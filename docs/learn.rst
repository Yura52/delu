.. _learn:

Learn
=====

How to install DeLU?
--------------------

For installation instructions, see :ref:`Installation <installation>`.

What is Reference?
----------------------

Reference (see the sidebar) provides API specification, tutorials and usage examples.
It is the main source of knowledge about the library.

How to learn DeLU?
------------------

DeLU, by design, is a "toolbox", not a "framework", so you can learn things and
start using them in almost any order. If you want to explore the library by yourself,
just browse the Reference.

However, if you want to quickly learn some basic things, then understanding of
`this classification task <https://github.com/Yura52/delu/blob/main/examples/mnist.py>`_
is enough. To achieve this, you need to walk through some specific parts of Reference
paying attention to *explanations* and *examples*, without diving into API details. Here
are the things to learn:

.. autosummary::
   :nosignatures:

   delu.random
   delu.cat
   delu.ProgressTracker
   delu.Timer
   delu.hardware.get_gpus_info

Congratulations! You are ready to apply DeLU in practice.
