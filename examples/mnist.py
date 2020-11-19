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

import zero


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
    parser.add_argument('-n', '--n-epochs', type=int, default=20)
    parser.add_argument('-p', '--early-stopping-patience', type=int, default=2)
    parser.add_argument('-s', '--seed', type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    zero.set_randomness(args.seed)
    model = nn.Linear(784, 10).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), 0.005, 0.9)

    def step(batch):
        X, y = batch
        return model(X.to(args.device)), y.to(args.device)

    def evaluate(loader):
        with zero.evaluate(model):
            logits, y = zero.concat(map(step, loader))
        y_pred = torch.argmax(logits, dim=1).to(y)
        return (y_pred == y).int().sum().item() / len(y)

    train_dataset, val_dataset = split_dataset(get_dataset(True), 0.8)
    test_dataset = get_dataset(False)
    stream = zero.Stream(DataLoader(train_dataset, batch_size=64, shuffle=True))
    val_loader = DataLoader(val_dataset, batch_size=8096)
    test_loader = DataLoader(test_dataset, batch_size=8096)

    timer = zero.Timer()
    progress = zero.ProgressTracker(args.early_stopping_patience, 0.005)
    best_model_path = 'model.pt'

    for epoch in stream.epochs(args.n_epochs, args.epoch_size):
        print(f'\nEpoch {stream.epoch} started (iterations passed: {stream.iteration})')
        timer.run()

        for batch in epoch:
            zero.learn(model, optimizer, F.cross_entropy, step, batch, True)

        timer.pause()
        accuracy = evaluate(val_loader)
        progress.update(accuracy)
        if progress.success:
            torch.save(model.state_dict(), best_model_path)

        msg = (
            f'Epoch {stream.epoch} finished. '
            f'Time elapsed: {zero.format_seconds(timer())}. '
            f'Validation accuracy: {accuracy:.4f}.'
        )
        if args.device.type == 'cuda':
            index = args.device.index or 0
            msg += f'\nGPU info: {zero.get_gpu_info()[index]}'
        print(msg)
        if progress.fail:
            break

    timer.pause()
    model.load_state_dict(torch.load(best_model_path))
    print(
        f'\nTraining stopped after the epoch {stream.epoch}.\n'
        f'Total training time: {zero.format_seconds(timer())}\n'
        f'Test accuracy: {evaluate(test_loader)}'
    )

    print('Freeing memory (for fun, not for profit) ...')
    del model, optimizer, step, evaluate
    zero.free_memory()
    print('\nDONE.')


if __name__ == '__main__':
    main()
