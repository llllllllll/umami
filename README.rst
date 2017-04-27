Umami
=====

Umami is a package to perform Bayesian optimization according to the
algorithms outlined in the paper:

| **Practical Bayesian Optimization of Machine Learning Algorithms**
| Jasper Snoek, Hugo Larochelle and Ryan P. Adams
| *Advances in Neural Information Processing Systems*, 2012

Umami is a fork of the GPLv3+ version of `Spearmint
<https://github.com/JasperSnoek/spearmint>`_. I named it Umami because I am bad
at naming projects.

I chose to fork Spearmint because development moved to an academic-use-only
license and the project seems to have died.

Goals
-----

- Port to Python 3 (only)
- Clean up the source.
- Prune code to a maintainable subset.
- Improve the user experience.

I have gotten Umami into a state where it actually runs (and optimizes!) local
Python tasks, but there is a lot of work left. I plan on porting to
Python 3. Right now it seems the biggest blocker is Weave which is used for
a single for loop which could be written with better usage Numpy or maybe Numba
if I can't figure that out.

While cleaning up the source code I plan to prune the features down to something
more manageable. I plan on only supporting local execution of Python files. If
you want to run a task in a different language you could write a little Python
file to wrap the calls to your language, Umami doesn't need to support this
directly.

I would like to improve the output of Umami. Currently it just dumps a bunch of
files in the experiment directory but this could be stored in a standard
location or format.

Usage
-----

Umami uses a ``config.json`` file which is mostly like Spearmint. Currently
Umami only supports Python and local execution so these options are ignored in
the ``config.json`` file.

Umami can be run on the command line with:

.. code-block::

   $ python -m umami /path/to/config.json [OPTIONS]

Example
~~~~~~~

Below is an example ``config.json`` which show different variable types and
shapes.

.. code-block:: json

   {
       "main-file"       : "run_task.py",
       "name"            : "mlp-regressor-all-solvers",
       "likelihood"      : "GAUSSIAN",
       "variables" : {
           "alpha" : {
               "type" : "FLOAT",
               "size" : 1,
               "min"  : 0.00001,
               "max"  : 0.009
           },
           "hidden_layer_sizes" : {
               "type" : "INT",
               "size" : 3,
               "min"  : 1,
               "max"  : 200
           },
           "solver": {
               "type": "ENUM",
               "size": 1,
               "options": ["lbfgs", "sgd", "adam"]
           },
           "activation": {
               "type": "ENUM",
               "size": 1,
               "options": ["identity", "logistic", "tanh", "relu"]
           }
       }
   }

The ``main-file`` must define a function ``main`` like:

.. code-block:: python

   def main(job_id, params):
       # train
       return error

``params`` is passed as a dictionary from variable name to values based on the
variable spaces defined in ``config.json``.
