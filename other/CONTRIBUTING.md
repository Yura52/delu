### Setup
1. Fork the repository.

2. Clone the repository
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

4. Set up a pre-commit hook (use "`git commit -n ...`" to avoid running it for WIP-like or non-code commits):
```bash
$ echo "#!/bin/sh

export PYTHONPATH=<REPOSITORY_ROOT>:$PYTHONPATH
export PATH=<CONDA_ENVIRONMENTS>/zero/bin:$PATH
make pre-commit
" > .git/hooks/pre-commit
```

5. Learn [Makefile](../Makefile). It contains shortcuts for running linters, code formatters, tests, the pre-commit hook and other useful commands.

### Checklist for you Pull Request
- [ ] `make pre-commit` succeeds
- [ ] `make coverage` shows that all new code is covered with tests
- [ ] docstrings are added

### Notes
- "# mypy: NaN" means either "I don't know how to make mypy happy" or "It is impossible to make mypy happy" or "Making mypy happy requires going crazy with type annotations"