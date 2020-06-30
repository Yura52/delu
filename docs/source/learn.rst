Learn
=====

How to install Zero?
--------------------

For installation instructions, see :ref:`Installation`.

What is Reference?
------------------

Reference (see the sidebar) is the main source of knowledge about the library. It
contains everything you may need to understand how to use specific tools: API
specification, tutorials and usage examples.

How to learn Zero?
------------------

If you just want to "browse" the library and see what it offers, then
:ref:`this page <zero>` is a good place to start (Zero, by design, is rather a
"toolbox" than a "framework", so you can learn things in almost any order).

However, if you want to quickly learn the basics and start using Zero in your research
and training pipelines, understanding of
`this classification task <https://github.com/Yura52/zero/blob/master/examples/mnist.py>`_
is enough. To achieve this, you need to walk through some specific parts of Reference
paying attention to *explanations* and *examples*, without diving into API details. Here
are the things to learn:

.. autosummary::

   zero.flow.Flow
   zero.metrics.Metric
   zero.optim
   zero.model.Eval
   zero.concat_dmap
   zero.progress.ProgressTracker
   zero.tensor.to_device
   zero.tensor.ibackward
   zero.random.set_seed_everywhere
   zero.time.Timer
   zero.hardware
   zero.io

Congratulations! You are ready to apply Zero in practice. You can also visit
:ref:`this page <zero>` and explore things that are not covered in the list above (
:ref:`zero.data <data>` and others).
