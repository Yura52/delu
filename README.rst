DeLU (Deep Learning Utilities)
==============================

`Documentation <https://yura52.github.io/delu>`_

.. __INCLUDE_0__

**DeLU is a simple PyTorch toolbox** consisting of the following parts:

- Extensions to PyTorch submodules:

  - ``delu`` ~ ``torch``
  - ``delu.cuda`` ~ ``torch.cuda``
  - ``delu.nn`` ~ ``torch.nn``
  - ``delu.random`` ~ ``torch.random``
  - ``delu.utils.data`` ~ ``torch.utils.data``

- ``delu.tools``: handy tools for common scenarios
  (e.g. for implementing training loops).
- ``delu.tabular``: tools for working on tabular data problems.

**Project status** 🧪 *Until the release of v0.1.0
(which is not guaranteed to happen), DeLU should be considered as experimental.
All changes are carefully documented in the GitHub releases.*

Installation
------------

Without ``delu.tabular``:

.. code-block:: none

    pip install delu

With ``delu.tabular`` (must be imported separately):

.. code-block:: none

    pip install delu[tabular]

Usage
-----

*DeLU is a toolbox, not a framework,
so you can learn things and start using them in any order.*

The "API & Examples" section on the website is the main source of knowledge about DeLU:
it provides usage examples, explanation and docstrings.

How to contribute
-----------------

- Use `issues <https://github.com/Yura52/delu/issues>`_
  to report bugs, share ideas and give feedback.
- For pull requests, see
  `CONTRIBUTING.md <https://github.com/Yura52/delu/blob/main/CONTRIBUTING.md>`_.
