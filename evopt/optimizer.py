"""High-level interface for black-box optimization algorithms.

This module provides a simplified entry point to the black-box optimization framework,
abstracting away the details of specific optimization algorithms. It offers a convenient
function-based interface for running optimization tasks with minimal setup.

The module supports different optimization algorithms (currently CMA-ES) and handles
both local and HPC-based parallel execution environments.

Example:
    Basic usage for a simple optimization problem:
    
    >>> from evopt import optimize
    >>> 
    >>> # Define parameter space
    >>> parameters = {'x': (-10, 10), 'y': (-5, 5)}
    >>> 
    >>> # Define evaluation function
    >>> def evaluate(params):
    >>>     return params['x']**2 + params['y']**2  # Simple quadratic function
    >>> 
    >>> # Run optimization with default settings
    >>> result = optimize(parameters, evaluate, batch_size=10, max_workers=4)
    >>> print(f"Best solution: {result.best_parameters}")
    >>> print(f"Best error: {result.final_error}")
"""

import os
from typing import Callable
from .cma_optimizer import CmaesOptimizer, OptimizationResults
from .directory_manager import DirectoryManager

def optimize(
    params: dict,
    evaluator: Callable,
    optimizer: str = 'cmaes',
    base_dir: str = None,
    dir_id: int = None,
    sigma_threshold: float = 0.1,
    batch_size: int = 16,
    start_epoch: int = None,
    verbose: bool = True,
    num_epochs: int = None,
    target_dict: dict = None,
    max_workers: int = 1,
    rand_seed: int = None,

    # HPC-specific parameters
    hpc_cores_per_worker: int = 1, 
    hpc_memory_gb_per_worker: int = 4,
    hpc_wall_time: str = "1:00:00",
    hpc_qos: str = None
) -> OptimizationResults:
    """Run an evolutionary optimization with the specified parameters and evaluator.
    
    This function provides a simplified interface to run evolutionary optimization
    algorithms without needing to directly instantiate optimizer classes. It handles
    directory management, optimizer selection, and cleanup of resources.
    
    Args:
        params: Parameter space definition as a dictionary where keys are parameter names
            and values are tuples of (min_value, max_value) bounds.
        evaluator: Function that evaluates a parameter set and returns either:
            - A single float value representing the error/fitness (lower is better)
            - A dictionary of observed values to be compared with target_dict
        optimizer: Name of the optimization algorithm to use. 
            Currently supported: 'cmaes'. Default is 'cmaes'.
        base_dir: Base directory for storing optimization results. If None,
            uses the current working directory. Default is None.
        dir_id: Specific identifier for this optimization run. If None, 
            a new ID will be automatically generated. Default is None.
        sigma_threshold: Convergence threshold for normalized sigma values.
            Lower values require more precise convergence. Default is 0.1.
        batch_size: Number of solutions to evaluate in each optimization epoch.
            Higher values provide more exploration but require more compute.
            Default is 16.
        start_epoch: Epoch number to resume optimization from. If None, starts
            a new optimization run. Default is None.
        verbose: Whether to print detailed progress information during optimization.
            Default is True.
        num_epochs: Maximum number of epochs to run. If None, runs until convergence
            criteria are met. Default is None.
        target_dict: Dictionary of target values to optimize towards. If provided,
            the evaluator should return a dictionary of observed values to be compared
            with these targets. Default is None.
        max_workers: Maximum number of parallel workers for solution evaluation.
            Default is 1 (no parallelization).
        rand_seed: Seed for random number generation. If None, uses dir_id as seed.
            Default is None.
        hpc_cores_per_worker: Number of CPU cores to allocate per worker in HPC
            environments. Default is 1.
        hpc_memory_gb_per_worker: Memory in GB to allocate per worker in HPC
            environments. Default is 4.
        hpc_wall_time: Maximum wall time for HPC jobs in format "DD:HH:MM:SS"
            or "HH:MM:SS". Default is "1:00:00" (1 hour).
        hpc_qos: Quality of Service specification for SLURM jobs. Default is None.
            
    Returns:
        OptimizationResults: A comprehensive results object containing the best
            parameters found, optimization history, and algorithm statistics.
            
    Raises:
        ValueError: If an unsupported optimizer name is provided.
        
    Examples:
        Basic optimization of a simple function:
        
        >>> def sphere_function(params):
        ...     return sum(value**2 for value in params.values())
        >>> 
        >>> params = {'x1': (-5, 5), 'x2': (-5, 5), 'x3': (-5, 5)}
        >>> result = optimize(params, sphere_function, batch_size=8, max_workers=2)
        >>> print(f"Best parameters: {result.best_parameters}")
        
        Optimization with target values (multi-objective):
        
        >>> def structural_simulator(params):
        ...     # Simulation returning multiple metrics
        ...     return {
        ...         'weight': params['height'] * params['width'] * params['length'] * 0.1,
        ...         'stress': 100 / (params['height'] * params['width']),
        ...         'deflection': 50 / (params['height']**2)
        ...     }
        >>> 
        >>> # Define target values
        >>> targets = {
        ...     'weight': 10.0,      # Target weight
        ...     'stress': (0, 50),   # Stress must be below 50 (hard constraint range)
        ...     'deflection': 0.5    # Target deflection
        ... }
        >>> 
        >>> params = {'height': (1, 5), 'width': (1, 10), 'length': (10, 50)}
        >>> result = optimize(params, structural_simulator, target_dict=targets, 
        ...                  max_workers=4, num_epochs=50)
    """

    directory_manager = DirectoryManager(
        base_dir = os.getcwd() if base_dir is None else base_dir,
        dir_id = dir_id
    )
    if optimizer.lower() == 'cmaes':
        optimizer_class = CmaesOptimizer
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    optimizer = optimizer_class(
        parameters=params,
        evaluator=evaluator,
        n_epochs=num_epochs,
        batch_size=batch_size,
        directory_manager=directory_manager,
        sigma_threshold=sigma_threshold,
        rand_seed=rand_seed if rand_seed is not None else dir_id,
        start_epoch=start_epoch,
        target_dict=target_dict,
        verbose=verbose,
        max_workers=int(max_workers),
        cores_per_worker=hpc_cores_per_worker,
        memory_gb_per_worker=hpc_memory_gb_per_worker,
        wall_time = hpc_wall_time,
        qos=hpc_qos
    )
    try:
        with directory_manager.logger:
            params = optimizer.optimize()
        return params
    finally:
        if hasattr(optimizer, 'cleanup'):
            optimizer.cleanup() # closes process workers