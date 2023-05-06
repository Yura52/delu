# Do Ctrl+F "delu" to see how DeLU is used.
# Run `python mnist.py --help` to see the documentation

# ruff: noqa: F821,E501

import shlex
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

import delu


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
    parser.add_argument('-n', '--n-epochs', type=int, default=20)
    parser.add_argument('-p', '--early-stopping-patience', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-c', '--from-checkpoint')
    parser.add_argument('--skip-download', action='store_true')
    return parser.parse_args()


def main():
    assert str(Path.cwd().absolute().resolve()).endswith(
        'delu/examples'
    ), 'Run this script from the "examples" directory'
    args = parse_args()
    assert args.n_epochs > 0

    if not args.skip_download:
        download_mnist()
    # Set seeds in all "standard" libraries: random, numpy, torch
    delu.random.seed(args.seed)
    model = nn.Linear(784, 10).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), 0.005, 0.9)

    def step(batch):
        # Change device and/or dtype for a collection of tensors.
        X, y = delu.to(batch, args.device)
        return model(X), y

    @torch.inference_mode()
    def evaluate(loader):
        model.eval()
        # Concatenate a sequence of tuples `(batch_logits, batch_y)` into a single tuple.
        logits, y = delu.cat(map(step, loader))
        y_pred = torch.argmax(logits, dim=1).to(y)
        return (y_pred == y).int().sum().item() / len(y)

    train_dataset, val_dataset = split_dataset(get_dataset(True), 0.8)
    test_dataset = get_dataset(False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8096)
    test_loader = DataLoader(test_dataset, batch_size=8096)

    # This thing measures time!
    timer = delu.Timer()
    # Early stopping to prevent overfitting.
    early_stopping = delu.EarlyStopping(args.early_stopping_patience, mode='max')
    best_val_accuracy = float('-inf')
    checkpoint_path = 'checkpoint.pt'
    if args.from_checkpoint:
        assert args.from_checkpoint != 'checkpoint.pt'
        assert Path(args.from_checkpoint).exists()
        checkpoint = torch.load(args.from_checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        timer = checkpoint['timer']
        early_stopping = checkpoint['early_stopping']
        best_val_accuracy = checkpoint['best_val_accuracy']
        # Restore the global random state for full reproducibility.
        delu.random.set_state(checkpoint['random_state'])
        print('Resuming from the checkpoint.\n')
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.n_epochs):
        print(f'\nEpoch {epoch} started')

        # or use `with timer:` instead of timer.run/pause
        timer.run()
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            F.cross_entropy(*step(batch)).backward()
            optimizer.step()
        timer.pause()

        accuracy = evaluate(val_loader)
        early_stopping.update(accuracy)
        if accuracy > best_val_accuracy:
            print('New best score!')
            best_val_accuracy = accuracy
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'timer': timer,
                    'early_stopping': early_stopping,
                    'best_val_accuracy': best_val_accuracy,
                    # Save the global random state.
                    'random_state': delu.random.get_state(),
                },
                checkpoint_path,
            )

        msg = (
            f'Epoch {epoch} finished. '
            f'Time elapsed: {timer}. '
            f'Validation accuracy: {accuracy:.4f}.'
        )
        print(msg)
        if early_stopping.should_stop():
            break

    timer.pause()
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    print(
        f'\nTraining stopped after the epoch {epoch}.\n'
        f'Total training time: {timer}\n'
        f'Test accuracy: {evaluate(test_loader)}'
    )

    print('Freeing memory (for fun, not for profit) ...')
    del model, optimizer, step, evaluate
    # A handy function for freeing GPU memory.
    delu.cuda.free_memory()
    print('\nDONE.')
    if not args.from_checkpoint:
        print(
            '\nNow, you can test if the last epochs can be reproduced when the '
            'training is resumed from the last available checkpoint. For that, run:\n'
            'cp checkpoint.pt a.pt\n'
            f'python {shlex.join(sys.argv)} -c a.pt'
        )


if __name__ == '__main__':
    main()
