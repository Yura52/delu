### Setup
1. Fork the repository.

2. Clone your fork
```bash
$ git clone <your fork link> 
$ cd delu
```

3. Create a virtual environment with python and the **minimum required versions** of all the depependendices listed in the "dependencies" section of [pyproject.toml](./pyproject.toml). Also, in this environment, run:
```bash
pip install -r requirements_dev.txt
```

4. Set up the pre-commit hook by putting the following content into `.git/hooks/pre-commit` (use "`git commit -n ...`" to avoid running the hook for unfinished work):
```bash
#!/bin/sh

export PYTHONPATH=<REPOSITORY_ROOT>:$PYTHONPATH
export PATH=<CONDA_ENVIRONMENTS>/delu/bin:$PATH
make pre-commit
```

5. Learn the [Makefile](../Makefile). It contains shortcuts for running linters, code formatters, tests, the pre-commit hook and other useful commands. **All commands must be run from the repository root.**

6. Implement changes and open a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

P.S. Please, do not update the "gh-pages" branch. As of now, the website updates are
performed manually and the process is not documented.
