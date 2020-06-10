### Setup
Clone the repository
```bash
$ git clone <your fork link> 
$ cd zero
```

Create a virtual environment, install packages:
```bash
$ conda create -n zero "python=3.6.*"
$ conda activate zero
$ pip install numpy pynvml
$ <install torch, see https://pytorch.org/get-started/locally>
$ pip install -r other/requirements_dev.txt
```

Set up a pre-commit hook (use "`git commit -n ...`" to avoid running it for WIP-like or non-code commits)
```bash
$ echo "#!/bin/sh

export PYTHONPATH="$REPOSITORY_ROOT:$PYTHONPATH"
export PATH="$CONDA_ENVIRONMENTS/zero/bin:$PATH"
make pre-commit
" > .git/hooks/pre-commit
```

### Learn Makefile
It contains shortcuts for running linters, code formatters, tests, the pre-commit hook and other useful commands.

### Notes
- "mypy: NaN" means either "I don't know how to make mypy happy" or "Making mypy happy requires going crazy with type annotations"
