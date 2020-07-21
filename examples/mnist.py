from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

try:
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
except ImportError:
    print(
        'Please, install the following packages to proceed:\n'
        '- torchvision (see https://pytorch.org/get-started)'
    )
    raise

from zero.all import (
    Eval,
    Stream,
    ProgressTracker,
    Timer,
    concat,
    dump_json,
    format_seconds,
    free_memory,
    get_gpu_info,
    learn,
    set_randomness,
    to_device,
)


def get_dataset(train):
    return MNIST(
        '.', train=train, transform=lambda x: ToTensor()(x).view(-1), download=True
    )


def split_dataset(dataset, ratio):
    size = len(dataset)
    first_size = int(ratio * size)
    return random_split(dataset, [first_size, size - first_size])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', default='cpu', type=torch.device)
    parser.add_argument('-e', '--epoch-size')
    parser.add_argument('-n', '--n-epoches', type=int)
    parser.add_argument('-p', '--early-stopping-patience', type=int, default=2)
    parser.add_argument('-s', '--seed', type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    set_randomness(args.seed)
    model = nn.Linear(784, 10).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), 0.005, 0.9)

    def step(batch):
        X, y = to_device(batch, args.device)
        return model(X), y

    def calculate_accuracy(loader):
        with Eval(model):
            logits, y = concat(map(step, loader))
        y_pred = torch.argmax(logits, dim=1).to(y)
        return (y_pred == y).int().sum().item() / len(y)

    train_dataset, val_dataset = split_dataset(get_dataset(True), 0.8)
    test_dataset = get_dataset(False)
    val_loader = DataLoader(val_dataset, batch_size=8096)
    test_loader = DataLoader(test_dataset, batch_size=8096)

    stream = Stream(DataLoader(train_dataset, batch_size=64, shuffle=True))
    timer = Timer()
    progress = ProgressTracker(args.early_stopping_patience, 0.005)
    best_model_path = 'model.pt'

    while not progress.fail and stream.increment_epoch(args.n_epoches):
        print(f'\nEpoch {stream.epoch} started')
        timer.run()

        for batch in stream.data(args.epoch_size):
            loss = learn(model, optimizer, F.cross_entropy, step, batch, True)[0]
            if stream.iteration % 100 == 0:
                print(f'Iteration: {stream.iteration} Train loss: {loss:.4f}')

        timer.pause()
        accuracy = calculate_accuracy(val_loader)
        progress.update(accuracy)
        if progress.success:
            torch.save(model.state_dict(), best_model_path)

        msg = (
            f'Epoch {stream.epoch} finished. '
            f'Time elapsed: {format_seconds(timer())}. '
            f'Validation accuracy: {accuracy:.4f}.'
        )
        if args.device.type == 'cuda':
            index = args.device.index or 0
            msg += f'\nGPU info: {get_gpu_info()[index]}'
        print(msg)

    timer.pause()
    print(
        f'\nTraining stopped after the epoch {stream.epoch}. '
        f'Total training time: {format_seconds(timer())}'
    )

    model.load_state_dict(torch.load(best_model_path))
    report = {'accuracy': calculate_accuracy(test_loader)}
    report['training_time'] = timer()
    report_path = 'report.json'
    dump_json(report, report_path, indent=4)
    print(f'The experiment report is saved to {report_path}')

    print('Freeing memory (for fun, not for profit) ...')
    del model, optimizer, step
    free_memory()

    print('\nDONE.')


if __name__ == '__main__':
    main()
