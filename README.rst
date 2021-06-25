Zero
====

.. raw:: html

  <img src="https://raw.githubusercontent.com/Yura52/zero/master/docs/images/logo.png" width="130px" style="text-align:center;display:block;">

.. __INCLUDE_0__

Zero is a general-purpose library for PyTorch users. Zero simplifies training loops,
facilitates reproducibility, helps with models evaluation and other typical Deep Learning
tasks. Zero is a toolbox, not a framework:

- your training loop stays the same (regardless of whether it is powered by a simple Python loop or by a specialized framework)
- you can start by using any single tool you need, there is no "central concept"
- you can replace tools from Zero with custom solutions at any moment

**NOTE:** Zero is tested (and battle-tested in research projects), but the interface is
not stable yet, so backward-incompatible changes in future releases are possible.

Overview
--------

- `Website <https://yura52.github.io/zero>`_
- `Learn Zero <https://yura52.github.io/zero/learn.html>`_
- `Classification task example (MNIST) <https://github.com/Yura52/zero/blob/master/examples/mnist.py>`_
- `The first release announcement <https://github.com/Yura52/zero/issues/21>`_
- `Discussion <https://github.com/Yura52/zero/discussions/26>`_
- `Code <https://github.com/Yura52/zero>`_

.. __INCLUDE_1__

Installation
------------

We recommend installing PyTorch **before** installing Zero.

.. code-block:: bash

    pip install libzero

**Dependencies:** see `pyproject.toml` in the repository.

How to contribute
-----------------

- See `issues <https://github.com/Yura52/zero/issues>`_, especially with the labels
  `"discussion" <https://github.com/Yura52/zero/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3Adiscussion>`_
  and `"help wanted" <https://github.com/Yura52/zero/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`_
- `Open issues <https://github.com/Yura52/zero/issues/new/choose>`_ with bugs, ideas and
  any other kind of feedback

If your contribution includes pull requests, see `CONTRIBUTING.md <https://github.com/Yura52/zero/blob/master/other/CONTRIBUTING.md>`_

How to cite
-----------

.. code-block:: none

    @article{gorishniy2020zero,
        title={Zero: a Zero-Overhead Library for PyTorch Users},
        author={Yury Gorishniy},
        journal={GitHub},
        volume={Yura52/zero},
        url={https://github.com/Yura52/zero},
        year={2020},
    }

Why "Zero"?
-----------

Zero aims to be `zero-overhead <https://isocpp.org/wiki/faq/big-picture#zero-overhead-principle>`_
in terms of *mental* overhead: solutions, provided by Zero, try to
be as minimal, intuitive and easy to learn, as possible. Well, all these things can be
pretty subjective, so don't take it too seriously :wink:
