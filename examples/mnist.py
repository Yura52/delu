# Run `python mnist.py --help` to see the documentation

import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

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


def download_mnist():
    path = Path('MNIST.tar.gz')
    if path.exists():
        path.unlink()
        shutil.rmtree('MNIST', ignore_errors=True)
    subprocess.run(['wget', 'www.di.ens.fr/~lelarge/MNIST.tar.gz'], check=True)
    subprocess.run(['tar', '-zxvf', 'MNIST.tar.gz'], check=True)


def get_dataset(train):
    return MNIST('.', train=train, transform=lambda x: ToTensor()(x).view(-1))


def split_dataset(dataset, ratio):
    size = len(dataset)
    first_size = int(ratio * size)
    return random_split(dataset, [first_size, size - first_size])  # type: ignore[code]


def parse_args():
    parser = ArgumentParser(epilog='Example: python mnist.py')
    parser.add_argument('-d', '--device', default='cpu', type=torch.device)
    parser.add_argument(
        '-e', '--epoch-size', type=int, help='Number of batches per epoch'
    )
    parser.add_argument('-n', '--n-epochs', type=int, default=20)
    parser.add_argument('-p', '--early-stopping-patience', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-c', '--from-checkpoint')
    return parser.parse_args()


def main():
    assert str(Path.cwd().absolute().resolve()).endswith(
        'zero/examples'
    ), 'Run this script from the "examples" directory'
    args = parse_args()

    download_mnist()
    zero.improve_reproducibility(args.seed)
    model = nn.Linear(784, 10).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), 0.005, 0.9)

    def step(batch):
        X, y = batch
        return model(X.to(args.device)), y.to(args.device)

    def evaluate(loader):
        with zero.evaluation(model):
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
    checkpoint_path = 'checkpoint.pt'
    if args.from_checkpoint:
        assert args.from_checkpoint != 'checkpoint.pt'
        assert Path(args.from_checkpoint).exists()
        checkpoint = torch.load(args.from_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        stream.load_state_dict(checkpoint['stream'])
        timer = checkpoint['timer']
        progress = checkpoint['progress']
        zero.random.set_state(checkpoint['random_state'])
        print('Resuming from the checkpoint.\n')

    for epoch in stream.epochs(args.n_epochs, args.epoch_size):
        print(f'\nEpoch {stream.epoch} started (iterations passed: {stream.iteration})')
        timer.run()

        for batch in epoch:
            model.train()
            optimizer.zero_grad()
            F.cross_entropy(*step(batch)).backward()
            optimizer.step()

        timer.pause()
        accuracy = evaluate(val_loader)
        progress.update(accuracy)
        if progress.success:
            print('New best score!')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'stream': stream.state_dict(),
                    'timer': timer,
                    'progress': progress,
                    'random_state': zero.random.get_state(),
                },
                checkpoint_path,
            )

        msg = (
            f'Epoch {stream.epoch} finished. '
            f'Time elapsed: {timer}. '
            f'Validation accuracy: {accuracy:.4f}.'
        )
        if args.device.type == 'cuda':
            index = args.device.index or 0
            msg += f'\nGPU info: {zero.hardware.get_gpus_info()["devices"][index]}'
        print(msg)
        if progress.fail:
            break

    timer.pause()
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    print(
        f'\nTraining stopped after the epoch {stream.epoch}.\n'
        f'Total training time: {timer}\n'
        f'Test accuracy: {evaluate(test_loader)}'
    )

    print('Freeing memory (for fun, not for profit) ...')
    del model, optimizer, step, evaluate
    zero.hardware.free_memory()
    print('\nDONE.')
    if not args.from_checkpoint:
        # TODO replace `join` with `shlex.join` in 3.8
        print(
            '\nNow, you can test if the last epochs can be reproduced when the '
            'training is resumed from the last available checkpoint. For that, run:\n'
            'cp checkpoint.pt a.pt\n'
            f'python {" ".join(sys.argv)} -c a.pt'
        )


if __name__ == '__main__':
    main()
