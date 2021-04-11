Learn
=====

How to install Zero?
--------------------

For installation instructions, see :ref:`Installation <installation>`.

What is Reference?
------------------

Reference (see the sidebar) is the main source of knowledge about the library. It
contains everything you may need to understand how to use specific tools: API
specification, tutorials and usage examples.

How to import Zero?
-------------------

Simply do :code:`import zero` and use what you need without accessing the submodules.
For example, :code:`zero.evaluation` is the correct way to use the function, while
:code:`zero.learning.evaluation` is not, because the structure of submodules is not
stable yet.

How to learn Zero?
------------------

Zero, by design, is rather a "toolbox" than a "framework", so you can learn things and
start using them in almost any order. If you want to explore the library by yourself,
:ref:`this page <zero>` is a good place to start.

However, if you want to quickly learn "the most important" things, then understanding of
`this classification task <https://github.com/Yura52/zero/blob/master/examples/mnist.py>`_
is enough. To achieve this, you need to walk through some specific parts of Reference
paying attention to *explanations* and *examples*, without diving into API details. Here
are the things to learn:

.. autosummary::

   zero.learning.evaluation
   zero.random.set_randomness
   zero.random.get_random_state
   zero.tools.ProgressTracker
   zero.tools.Timer
   zero.hardware.get_gpus_info
   zero.data.concat

Congratulations! You are ready to apply Zero in practice. You can also visit
:ref:`this page <zero>` and explore things that are not covered in the list above.
