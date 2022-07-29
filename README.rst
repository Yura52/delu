DeLU (Deep Learning Utilities)
==============================

.. raw:: html

  <img src="https://raw.githubusercontent.com/Yura52/delu/main/docs/images/logo.png" width="130px" style="text-align:center;display:block;">

.. __INCLUDE_0__

DeLU is a general-purpose library for PyTorch users. DeLU simplifies training loops,
facilitates reproducibility, helps with models evaluation and other typical Deep Learning
tasks. DeLU is a toolbox, not a framework:

- your training loop stays the same (regardless of whether it is powered by a simple Python loop or by a specialized framework)
- you can start by using any single tool you need, there is no "central concept"
- you can replace tools from DeLU with custom solutions at any moment

**NOTE:** DeLU is tested (and battle-tested in research projects), but the interface is
not stable yet, so backward-incompatible changes in future releases are possible.

Overview
--------

- `Website <https://yura52.github.io/delu>`_
- `Learn DeLU <https://yura52.github.io/delu/stable/learn>`_
- `Classification task example (MNIST) <https://github.com/Yura52/delu/blob/main/examples/mnist.py>`_
- `Code <https://github.com/Yura52/delu>`_

.. __INCLUDE_1__

Installation
------------

.. code-block:: bash

    pip install delu

**Dependencies:** see `pyproject.toml` in the repository.

How to contribute
-----------------

- Help to resolve `issues <https://github.com/Yura52/delu/issues>`_
- Report `bugs and issues <https://github.com/Yura52/delu/issues/new/choose>`_
- Post `questions, ideas and feedback <https://github.com/Yura52/delu/discussions/new>`_

If your contribution includes pull requests, see `CONTRIBUTING.md <https://github.com/Yura52/delu/blob/main/other/CONTRIBUTING.md>`_

How to cite
-----------

.. code-block:: none

    @article{gorishniy2020delu,
        title={DeLU: a Lightweight Toolbox for Deep Learning Researchers and Practitioners},
        author={Yury Gorishniy},
        journal={GitHub},
        volume={Yura52/delu},
        url={https://github.com/Yura52/delu},
        year={2020},
    }
