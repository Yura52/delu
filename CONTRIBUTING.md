### Setup
1. Fork the repository.

2. Clone your fork
```bash
$ git clone <your fork link> 
$ cd zero
```

3. Create a virtual environment, install packages:
```bash
$ conda create -n zero "python=3.7.*"
$ conda activate zero
$ pip install "numpy==1.17.*" "pynvml==8.0.*"
$ <install torch==1.6.*, see https://pytorch.org/get-started/locally>
$ pip install -r other/requirements_dev.txt
```

4. Set up a pre-commit hook (use "`git commit -n ...`" to avoid running the hook for unfinished work):
```bash
$ echo "#!/bin/sh

export PYTHONPATH=<REPOSITORY_ROOT>:$PYTHONPATH
export PATH=<CONDA_ENVIRONMENTS>/zero/bin:$PATH
make pre-commit
" > .git/hooks/pre-commit
```

5. Learn the [Makefile](../Makefile). It contains shortcuts for running linters, code formatters, tests, the pre-commit hook and other useful commands. **All commands must be run from the repository root.**

6. Implement changes and open a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

P.S. Please, do not update the "gh-pages" branch. As of now, the website updates are
performed manually and the process is not documented.
