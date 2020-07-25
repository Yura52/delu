Zero
====

.. __INCLUDE_0__

Zero is a general-purpose library for PyTorch users. Zero:

- simplifies training loop, models evaluation, models application and other typical Deep
  Learning tasks
- provides a collection of "building blocks" and leaves code organization to you
- can be used on its own or together with PyTorch frameworks such as
  `Ignite <https://github.com/pytorch/ignite>`_,
  `Lightning <https://github.com/PytorchLightning/pytorch-lightning>`_,
  `Catalyst <https://github.com/catalyst-team/catalyst>`_ and
  `others <https://pytorch.org/ecosystem>`_

**NOT READY FOR PRODUCTION USAGE.** Zero is tested, but not battle-tested. You can give
it a try in non-mission-critical research.

Overview
--------

- `Website <https://yura52.github.io/zero>`_
- `Code <https://github.com/Yura52/zero>`_
- `Learn Zero <https://yura52.github.io/zero/learn.html>`_
- `Classification task example (MNIST) <https://github.com/Yura52/zero/blob/master/examples/mnist.py>`_
- Discussions:

  - `Issue on GitHub <https://github.com/Yura52/zero/issues/21>`_
  - (coming soon) Post on Medium
  - (coming soon) Discussion on Reddit
  - (coming soon) Discussion on Hacker News

.. __INCLUDE_1__

Installation
------------

If you plan to use the GPU version of PyTorch, install it **before** installing Zero
(otherwise, the CPU version will be installed together with Zero).

.. code-block:: bash

    $ pip install libzero

Dependencies
^^^^^^^^^^^^

- Python >= 3.6
- NumPy >= 1.17
- PyTorch >= 1.3 (CPU or CUDA >= 10.1)
- pynvml >= 8.0

How to contribute
-----------------

- See `issues <https://github.com/Yura52/zero/issues>`_, especially with the labels
  `"discussion" <https://github.com/Yura52/zero/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3Adiscussion>`_
  and `"help wanted" <https://github.com/Yura52/zero/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`_
- `Open issues <https://github.com/Yura52/zero/issues/new/choose>`_ with bugs, ideas and
  any other kind of feedback

If your contribution includes pull requests, see `CONTRIBUTING.md <https://github.com/Yura52/zero/blob/master/other/CONTRIBUTING.md>`_.

Why "Zero"?
-----------

Zero aims to be `zero-overhead <https://isocpp.org/wiki/faq/big-picture#zero-overhead-principle>`_
in terms of *mental* overhead: solutions, provided by Zero, try to
be as minimal, intuitive and easy to learn, as possible. Well, all these things can be
pretty subjective, so don't take it too seriously :wink:
