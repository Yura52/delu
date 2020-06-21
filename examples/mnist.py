from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
try:
    from sklearn.metrics import classification_report
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
except ImportError:
    print(
        'Please, install the following packages to proceed:\n'
        '- scikit-learn\n'
        '- torchvision (see https://pytorch.org/get-started)'
    )
    raise

from zero.all import (
    Flow,  # zero.flow
    free_memory, get_gpu_info, to_device,  # zero.hardware
    dump_json,  # zero.io
    concat, dmap,  # zero.map_concat
    Metric,  # zero.metrics
    SGD,  # zero.optim
    ProgressTracker,  # zero.progress
    set_seed_everywhere,  # zero.random
    Timer, format_seconds,  # zero.time
    Eval, ibackward,  # zero.training
)


class Accuracy(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_objects = 0
        self.n_correct = 0

    def update(self, logits, y):
        y_pred = logits.argmax(dim=1)
        self.n_objects += len(y)
        self.n_correct += (y_pred == y).sum().item()

    def compute(self):
        assert self.n_objects
        return self.n_correct / self.n_objects


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
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-e', '--epoch-size')
    parser.add_argument('-n', '--n-epoches', type=int)
    parser.add_argument('-p', '--early-stopping-patience', type=int, default=2)
    parser.add_argument('-s', '--seed', type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed_everywhere(args.seed)
    device = torch.device(args.device)
    model = nn.Linear(784, 10).to(device)
    optimizer = SGD(model.parameters(), 0.005, 0.9)
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy()

    train_dataset, val_dataset = split_dataset(get_dataset(True), 0.8)
    test_dataset = get_dataset(False)
    val_loader = DataLoader(val_dataset, batch_size=8096)
    test_loader = DataLoader(test_dataset, batch_size=8096)

    flow = Flow(DataLoader(train_dataset, batch_size=64, shuffle=True))
    timer = Timer()
    progress = ProgressTracker(args.early_stopping_patience, 0.005)
    best_model_path = 'model.pt'

    def step(batch):
        X, y = to_device(batch, device)
        return model(X), y  # noqa

    while not progress.fail and flow.increment_epoch(args.n_epoches):
        print(f'\nEpoch {flow.epoch} started')
        timer.start()

        for batch in flow.data(args.epoch_size):
            model.train()
            with optimizer:
                loss = ibackward(loss_fn(*step(batch)))
            if flow.iteration % 100 == 0:
                print(f'Iteration: {flow.iteration} Train loss: {loss:.3f}')

        timer.stop()
        with Eval(model), metric_fn:
            for batch in val_loader:
                metric_fn.update(*step(batch))
            score = metric_fn.compute()
        progress.update(score)
        if progress.success:
            torch.save(model.state_dict(), best_model_path)

        msg = (
            f'Epoch {flow.epoch} finished. '
            f'Time elapsed: {format_seconds(timer())}. '
            f'Validation accuracy: {score:.3f}.'
        )
        if device.type == 'cuda':
            index = device.index or 0
            msg += f'\nGPU info: {get_gpu_info()[index]}'
        print(msg)

    timer.stop()
    print(
        f'\nTraining stopped after the epoch {flow.epoch}. '
        f'Total training time: {format_seconds(timer())}'
    )

    model.load_state_dict(torch.load(best_model_path))
    with Eval(model):
        logits, y_pred = concat(dmap(step, test_loader, None, 'cpu'))
    report = classification_report(
        logits.argmax(dim=1).numpy(), y_pred.numpy(), output_dict=True
    )
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
