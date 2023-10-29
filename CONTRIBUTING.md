# Contributing

This document describes
how to set up the environment and make pull requests in this project.

Fork the repository and clone the fork:

```
git clone <your fork link> 
cd delu
```

Create the virtual environment
(replace `micromamba` with `mamba` or `conda` if needed):

```
micromamba create -f environment-dev.yaml
```

Set up the pre-commit hook by putting the following content into
`.git/hooks/pre-commit`
(use "`git commit -n ...`" to avoid running the hook for unfinished work):

```
#!/bin/sh

export PYTHONPATH=<REPOSITORY_ROOT>:$PYTHONPATH
export PATH=<CONDA_ENVIRONMENTS>/delu/bin:$PATH
make pre-commit
```

Check out the available commands in the [Makefile](../Makefile).
**All commands must be run from the repository root.**

Implement changes and open a
[pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

P.S. Please, do not update the "gh-pages" branch. As of now, the website updates are
performed manually and the process is not documented.
