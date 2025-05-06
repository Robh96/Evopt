Sampling (:py:mod:`evopt.sampler`)
==================================

The sampler module provides functionality for exploring parameter spaces using Sobol sequences,
which generate well-distributed points throughout the parameter space. This is useful for
understanding the landscape of your problem before optimization or for gathering training
data for surrogate models.

Functions
---------

.. currentmodule:: evopt.sampler

.. autofunction:: sample

   Core Functionality
   ~~~~~~~~~~~~~~~~~~

   This function enables efficient parameter space exploration. It supports:

   - Parameter space definition with min/max bounds
   - Sobol sequence generation for quasi-random sampling
   - Parallel evaluation of samples
   - Multi-objective evaluation with target dictionaries
   - CSV output for analysis
   - Directory organization similar to the optimization module

   Key Features
   ~~~~~~~~~~~~

   - **Quasi-Random Sampling**: Uses Sobol sequences to ensure better coverage of the parameter space than purely random sampling
   - **Parallelization**: Supports multi-core and multi-node evaluation of samples
   - **Consistent Storage**: Results are stored in a structured directory format compatible with plotting tools
   - **Target Support**: Works with the same target dictionary format as the optimizer module
   - **Resumability**: Can resume from interrupted sampling runs by tracking completed samples

   Usage Examples
   ~~~~~~~~~~~~~~

   Basic sampling of a parameter space::

       results = evopt.sample(
           params={'x': (-5, 5), 'y': (-5, 5)},
           evaluator=my_function,
           n_samples=64
       )

   Parallel sampling with targets::

       results = evopt.sample(
           params=params_dict,
           evaluator=my_evaluator,
           target_dict=targets,
           n_samples=100,
           max_workers=4,
           cores_per_worker=2
       )
