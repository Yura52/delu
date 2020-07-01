### Setup
1. Fork the repository.

2. Clone your fork
```bash
$ git clone <your fork link> 
$ cd zero
```

3. Create a virtual environment, install packages:
```bash
$ conda create -n zero "python=3.6.*"
$ conda activate zero
$ pip install "numpy==1.18.*" "pynvml==8.0.*"
$ <install torch==1.5.*, see https://pytorch.org/get-started/locally>
$ pip install -r other/requirements_dev.txt
```

4. Set up a pre-commit hook (use "`git commit -n ...`" to avoid running the hook for WIP-like commits):
```bash
$ echo "#!/bin/sh

export PYTHONPATH=<REPOSITORY_ROOT>:$PYTHONPATH
export PATH=<CONDA_ENVIRONMENTS>/zero/bin:$PATH
make pre-commit
" > .git/hooks/pre-commit
```

5. Learn the [Makefile](../Makefile). It contains shortcuts for running linters, code formatters, tests, the pre-commit hook and other useful commands. **All commands must be run from the project root.**

### Notes
- "# mypy: NaN" means either "I don't know how to make mypy happy" or "It is impossible to make mypy happy" or "Making mypy happy requires going crazy with type annotations"
