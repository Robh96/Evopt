.. evopt documentation master file, created by
   sphinx-quickstart on Tue Mar 11 14:06:04 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

evopt: user friendly black-box numerical optimization
================================================================

.. image:: https://raw.githubusercontent.com/robh96/evopt/main/images/cover_img.png
   :alt: Optimization of the two parameter Ackley function
   :width: 450
   :align: center



**evopt** is a Python package for efficient parameter optimization using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm. It provides a straightforward way to find optimal parameters for complex problems, especially when the relationship between parameters and outcomes isn't easily calculable.

Key Features
------------

- **Simple interface** - Optimize complex problems with minimal code
- **CMA-ES algorithm** - Powerful, gradient-free optimization
- **Multi-objective optimization** - Handle multiple targets with hard and soft constraints
- **Result visualization** - Built-in plotting for optimization convergence and parameter relationships
- **Checkpointing** - Resume interrupted optimizations from saved states
- **Parallel evaluation** - Speed up optimization with multi-core processing
- **HPC support** - Run on high-performance computing clusters with SLURM integration

Installation
------------

.. code-block:: bash

   pip install evopt

Quick Start
-----------

.. code-block:: python

   import evopt
   
   # Define parameter space
   params = {
       'x': (-5, 5),
       'y': (-5, 5),
   }
   
   # Define evaluator function
   def evaluator(params):
       x, y = params['x'], params['y']
       return (1 - x)**2 + 100 * (y - x**2)**2  # Rosenbrock function
   
   # Run optimization
   results = evopt.optimise(
       params=params,
       evaluator=evaluator,
       batch_size=8
   )
   
   # Print best parameters found
   print(f"Best parameters: {results.best_parameters}")
   print(f"Final error: {results.final_error}")

Use Cases
---------

* **Simulation Calibration**: Find parameters that make simulations match reality
* **Engineering Design**: Optimize dimensions, materials, and other design variables
* **Machine Learning**: Tune hyperparameters for optimal model performance
* **Scientific Modeling**: Estimate parameters in complex physical models

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   getting started
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Public API:
   
   api/public/optimizer
   api/public/plotting

.. toctree::
   :maxdepth: 2
   :caption: Internal API:
   :hidden:
   
   api/internal/base_optimizer
   api/internal/cma_optimizer
   api/internal/directory_manager
   api/internal/loss
   api/internal/slurm_executor
   api/internal/utils

Indices and Search
------------------

* :ref:`genindex`
* :ref:`search`