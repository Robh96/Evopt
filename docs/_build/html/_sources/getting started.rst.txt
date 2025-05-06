Getting Started
===============

This guide will help you get started with ``evopt``, a user-friendly black-box exploration and optimization library.

Installation
------------

You can install ``evopt`` using pip:

.. code-block:: bash

   pip install evopt

Basic Concepts
--------------

Before using ``evopt``, it's helpful to understand a few key concepts:

* **Parameters**: These are the values you want to optimize, defined with minimum and maximum bounds.
* **Evaluator**: A function that takes parameter values and returns an error value (lower is better) or a dictionary of observed values.
* **Targets**: Desired values or constraints that your optimization should achieve. These are used for multi-objective optimization.
* **Optimization**: The process of finding parameter values that minimize error or best satisfy multiple objectives.

Types of Optimization
~~~~~~~~~~~~~~~~~~~~~

``evopt`` supports two main optimization approaches:

1. **Single-Objective**: Your evaluator returns a single error value to minimize.
2. **Multi-Objective**: Your evaluator returns a dictionary of values that are compared against target values.

With multi-objective optimization, you can specify:

* **Hard Constraints**: Requirements that must be satisfied (e.g., stress below safety limit)
* **Soft Objectives**: Goals to optimize toward (e.g., minimize weight)

Parameter Space Sampling
~~~~~~~~~~~~~~~~~~~~~~~~

Before performing an optimization, you may want to explore your parameter space to:

* Understand the landscape of your problem
* Identify promising regions for optimization
* Test your evaluator function on diverse inputs
* Gather training data for surrogate models

``evopt`` provides efficient sampling using Sobol sequences, which create well-distributed points throughout your parameter space.

.. code-block:: python

   import evopt
   
   # Define your parameter space
   params = {
       'x': (-5, 5),
       'y': (-5, 5)
   }
   
   # Define your evaluation function
   def my_evaluator(param_dict):
       x = param_dict['x']
       y = param_dict['y']
       return x**2 + y**2
   
   # Generate and evaluate 32 Sobol samples
   results = evopt.sample(
       params=params,
       evaluator=my_evaluator,
       n_samples=32,
       verbose=True
   )

Unlike optimization, sampling:
 
* Doesn't attempt to converge toward better solutions
* Provides consistent coverage of the entire parameter space
* Can be used independently or as a precursor to optimization
* More effective when chaining with symbolic regression or surrogate modeling

Your First Optimization
-----------------------

Here's a simple example to help you get started with ``evopt``:

.. code-block:: python

   import evopt
   
   # Step 1: Define your parameter space
   params = {
       'x': (-5, 5),  # Parameter 'x' can range from -5 to 5
       'y': (-5, 5)   # Parameter 'y' can range from -5 to 5
   }
   
   # Step 2: Create an evaluator function
   def my_evaluator(param_dict):
       x = param_dict['x']
       y = param_dict['y']
       # Simple quadratic function to minimize
       return x**2 + y**2
   
   # Step 3: Run the optimization
   results = evopt.optimize(
       params=params,
       evaluator=my_evaluator,
       batch_size=8,     # Number of solutions to evaluate per iteration
       max_workers=2     # Number of parallel evaluations
   )
   
   # Step 4: Examine the results
   print(f"Best parameters found: {results.best_parameters}")
   print(f"Best error value: {results.final_error}")

Multi-Objective Example
-----------------------

Here's how to use targets for multi-objective optimization:

.. code-block:: python

   import evopt
   
   # Define parameter space
   params = {
       'height': (1, 10),
       'width': (1, 10)
   }
   
   # Create evaluator that returns multiple metrics
   def beam_evaluator(params):
       h = params['height']
       w = params['width']
       return {
           'weight': h * w,              # We want to minimize weight
           'stress': 100 / (h * w),      # Stress must stay below a threshold
           'deflection': 200 / (h**2 * w)  # Deflection should be near target
       }
   
   # Define target values and constraints
   targets = {
       'weight': {'value': 10, 'hard': False},   # Soft objective: minimize weight
       'stress': (0, 50),                        # Hard constraint: stress < 50
       'deflection': 1.0                         # Hard constraint: close to 1.0
   }
   
   # Run optimization with targets
   results = evopt.optimize(
       params=params,
       evaluator=beam_evaluator,
       target_dict=targets,
       batch_size=8,
       max_workers=2
   )
   
   print(f"Best parameters: {results.best_parameters}")

Understanding the Output
------------------------

When you run the optimization, you'll see output like this:

.. code-block:: text

   Starting new CMA-ES run in directory ./evolve_0
   Epoch 0 | (1/8) | Params: [3.241, -1.569] | Error: 12.837
   Epoch 0 | (2/8) | Params: [-2.134, 2.845] | Error: 12.526
   ...
   Epoch 0 | Mean Error: 13.542 | Sigma Error: 3.892
   Epoch 0 | Mean Parameters: [0.241, 0.358] | Sigma parameters: [2.513, 2.498]
   ...
   Terminating after meeting termination criteria at epoch 15.

This shows:

* **Epoch**: The current iteration of the optimization algorithm
* **Parameters**: The values being tested
* **Error**: The result from your evaluator function
* **Statistics**: Summary statistics for each epoch
* **Termination**: When and why the optimization ended

Examining Results
-----------------

After optimization completes, you can access:

* **Best Parameters**: ``results.best_parameters`` - A dictionary of the best parameter values found
* **Final Error**: ``results.final_error`` - The lowest error value achieved
* **History**: Various history attributes like ``results.mean_error_history`` to see how the optimization progressed

Visualizing Results
-------------------

``evopt`` makes it easy to visualize your results:

.. code-block:: python

   # Path to your optimization results directory
   evolve_dir = "./evolve_0"
   
   # Plot convergence over time
   evopt.Plotting.plot_epochs(evolve_dir_path=evolve_dir)
   
   # Plot parameter relationships
   evopt.Plotting.plot_vars(
       evolve_dir_path=evolve_dir,
       x="x",
       y="y",
       cval="error"
   )

Next Steps
----------

Now that you understand the basics, you can:

* Try optimizing your own functions
* Explore multi-objective optimization with target dictionaries
* Use checkpointing for long-running optimizations
* Scale up to high-performance computing environments

See the :ref:`tutorials` section for more detailed examples.