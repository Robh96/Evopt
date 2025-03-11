"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation for optimization.

This module provides an implementation of the CMA-ES algorithm for black-box optimization
of non-linear or non-convex continuous optimization problems. CMA-ES is particularly
effective for problems that are ill-conditioned, have multiple local optima, or where
gradient information is not available.

The implementation extends the BaseOptimizer class and leverages the pycma library
for core CMA-ES functionality while handling parameter management, parallel evaluation,
convergence detection, and result logging.

Example:
    Basic usage with a simple objective function:
    
    >>> from evopt import CmaesOptimizer, DirectoryManager
    >>> parameters = {'x': (-10, 10), 'y': (-5, 5)}
    >>> def evaluator(params):
    >>>     return params['x']**2 + params['y']**2  # Simple quadratic function
    >>> dir_manager = DirectoryManager('./optimization_results')
    >>> optimizer = CmaesOptimizer(
    ...     parameters=parameters,
    ...     evaluator=evaluator,
    ...     batch_size=10,
    ...     directory_manager=dir_manager,
    ...     n_epochs=50
    ... )
    >>> results = optimizer.optimize()
    >>> print(f"Best parameters: {results.best_parameters}")
    >>> print(f"Final error: {results.final_error}")
"""

import cma
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from .base_optimizer import BaseOptimizer


class CmaesOptimizer(BaseOptimizer):
    """Optimization using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.
    
    This class implements the CMA-ES algorithm for black-box optimization of non-linear, 
    non-convex continuous optimization problems. CMA-ES adapts a multivariate normal 
    distribution to sample increasingly optimal solutions, using the covariance matrix
    to capture parameter dependencies and step-size adaptation for efficient convergence.
    
    The implementation uses the pycma library for core functionality while adding support
    for parallel evaluation, checkpoint management, and comprehensive result tracking.
    
    Attributes:
        es (cma.CMAEvolutionStrategy): The underlying CMA-ES optimizer instance.
        
    Note:
        This class inherits from BaseOptimizer and implements the required abstract methods.
        See BaseOptimizer for inherited attributes and methods.
        
    Example:
        >>> from evopt import CmaesOptimizer, DirectoryManager
        >>> # Define parameter space with bounds
        >>> parameters = {'length': (10, 100), 'width': (5, 50), 'height': (1, 10)}
        >>> # Define evaluator function (lower is better)
        >>> def evaluator(params):
        ...     volume = params['length'] * params['width'] * params['height']
        ...     return 1000 - volume  # Maximize volume by minimizing this value
        >>> 
        >>> # Set up optimizer
        >>> dir_manager = DirectoryManager('./cmaes_results')
        >>> optimizer = CmaesOptimizer(
        ...     parameters=parameters,
        ...     evaluator=evaluator,
        ...     batch_size=16,
        ...     directory_manager=dir_manager,
        ...     sigma_threshold=0.05,  # Convergence threshold
        ...     max_workers=4,  # Parallel evaluation
        ...     n_epochs=100  # Maximum epochs
        ... )
        >>> 
        >>> # Run optimization
        >>> results = optimizer.optimize()
        >>> 
        >>> # Access results
        >>> print(f"Best parameters: {results.best_parameters}")
        >>> print(f"Best fitness: {results.final_error}")
    """

    def setup_opt(self, epoch: int = None) -> None:
        """Set up the CMA-ES optimizer instance.
        
        Initializes or restores the CMA-ES optimizer either by creating a new instance
        or loading a checkpoint from a previous run. This method configures the CMA-ES
        algorithm with appropriate parameters and bounds.
        
        Args:
            epoch: The epoch number to resume from if restoring from a checkpoint.
                If None, starts a new optimization run. Default is None.
                
        Returns:
            None
            
        Note:
            This method stores the CMA-ES instance in self.es and updates the
            current_epoch counter.
            
        Example:
            >>> # Start a new optimization
            >>> optimizer.setup_opt()
            >>> 
            >>> # Resume from epoch 10
            >>> optimizer.setup_opt(epoch=10)
        """

        es = self.dir_manager.load_checkpoint(epoch)
        if es is None:
            opts = {
                'maxiter': self.n_epochs if self.n_epochs is not None else 1000000, # large number
                'seed': self.rand_seed,
                'popsize': self.batch_size,
                'bounds': [list(bound) for bound in zip(*self.norm_bounds)],
                'verbose': -9 # Silence most output, we handle our own reporting
            }
            warnings.simplefilter("ignore", UserWarning)
            es = cma.CMAEvolutionStrategy(self.init_params, 1.0, opts)
            if self.verbose:
                print(f"Starting new CMAES run in directory {self.dir_manager.evolve_dir}")
        elif self.verbose:
            print(f"Continuing CMAES run from epoch {es.countiter} in directory {self.dir_manager.evolve_dir}")
        self.es = es
        self.current_epoch = self.es.countiter
        
    def check_termination(self) -> bool:
        """Check if optimization termination criteria are met.
        
        Determines whether the optimization should terminate based on either:

        1. Convergence: All parameters' normalized standard deviations have fallen
           below the threshold, indicating the algorithm has converged
        2. Maximum epochs: The specified maximum number of epochs has been reached
        
        The method uses the standard deviations from the CMA-ES covariance matrix
        as a direct measure of search space exploration.
        
        Returns:
            bool: True if termination criteria are met (either convergence or
                maximum epochs reached), False otherwise.
                
        Example:
            >>> while not optimizer.check_termination():
            ...     # Run one iteration
            ...     solutions = optimizer.es.ask(optimizer.batch_size)
            ...     errors = optimizer.process_batch(solutions)
            ...     optimizer.es.tell(solutions, errors)
        """

        #sigmas = np.array([v[-1] for p, v in self.norm_sigmas.items() if v]) # old implementation
        sigmas = self.es.sigma * np.sqrt(np.diag(self.es.C))
        if len(sigmas) == 0:
            return False
        
        sigma_check = np.all(sigmas < self.sigma_threshold)
        epoch_check = self.n_epochs is not None and self.current_epoch >= self.n_epochs
        return sigma_check or epoch_check
    

    def optimize(self) -> 'OptimizationResults':
        """Run the CMA-ES optimization process until termination.
        
        Executes the complete CMA-ES optimization loop, handling:
        
        1. Setup/initialization of the optimizer
        2. Generation of candidate solutions
        3. Evaluation of solutions (potentially in parallel)
        4. Updating the internal state of the algorithm
        5. Checkpointing for resumability
        6. Termination detection
        7. Results compilation
        
        The method continues until either convergence criteria are met or
        the maximum number of epochs is reached.
        
        Returns:
            OptimizationResults: A comprehensive results object containing the best
                parameters found, optimization history, and algorithm-specific data.
                
        Example:
            >>> # Run the full optimization process
            >>> results = optimizer.optimize()
            >>> 
            >>> # Extract best results
            >>> best_params = results.best_parameters
            >>> error = results.final_error
            >>> 
            >>> # Analyze convergence behavior
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(results.mean_error_history)
            >>> plt.title("Error Convergence")
            >>> plt.xlabel("Epoch")
            >>> plt.ylabel("Error")
            >>> plt.show()
        """

        self.setup_opt(epoch=self.start_epoch)
        
        while not self.check_termination():
            solutions = self.es.ask(self.batch_size)
            errors = self.process_batch(solutions)
            self.es.tell(solutions, errors)
            self.es.disp()
            self.dir_manager.save_checkpoint(self.es, self.es.countiter)
            self.current_epoch = self.es.countiter
        
        if self.verbose:
            if self.n_epochs is not None and self.current_epoch >= self.n_epochs:
                termination_reason = "Maximum epochs reached"
                print(f"Terminating after reaching maximum epochs ({self.n_epochs}).")
            else:
                termination_reason = "Termination criteria met"
                print(f"Terminating after meeting termination criteria at epoch {self.current_epoch}.")
        
        # Create and return comprehensive results object
        results = OptimizationResults(
            best_parameters={p: float(v[-1]) for p, v in self._mean_params_history.items()},
            final_error=float(self._mean_error_history[-1]),
            mean_error_history=[float(x) for x in self._mean_error_history],
            sigma_error_history=[float(x) for x in self._sigma_error_history],
            mean_params_history={p: [float(x) for x in v] for p, v in self._mean_params_history.items()},
            sigma_params_history={p: [float(x) for x in v] for p, v in self._sigma_params_history.items()},
            norm_sigmas_history={p: [float(x) for x in v] for p, v in self._norm_sigmas_history.items()},
            mean_target_history={k: [float(x) for x in v] for k, v in self._mean_target_history.items()} 
                            if hasattr(self, '_mean_target_history') and self._mean_target_history is not None 
                            else None,
            sigma_target_history={k: [float(x) for x in v] for k, v in self._sigma_target_history.items()} 
                                if hasattr(self, '_sigma_target_history') and self._sigma_target_history is not None 
                                else None,
            epochs_completed=int(self.current_epoch),
            terminated_reason=termination_reason,
            cmaes_sigma=float(self.es.sigma),
            cmaes_C=self.es.C.copy() if hasattr(self.es, 'C') else None
        )
    
        return results
    

@dataclass
class OptimizationResults:
    """Container for comprehensive optimization results.
    
    This dataclass stores all results from a completed optimization run, including
    the best parameters found, error values, complete optimization history, 
    and algorithm-specific data. All numeric values are stored as native Python types
    for better serialization compatibility.
    
    Attributes:
        best_parameters: Dictionary of the best parameter values found, with parameter
            names as keys and their optimized values as values.
        final_error: The error/fitness value of the best solution.
        mean_error_history: List of mean error values for each epoch.
        sigma_error_history: List of error standard deviations for each epoch.
        mean_params_history: Dictionary of parameter means over time, with parameter 
            names as keys and lists of their mean values as values.
        sigma_params_history: Dictionary of parameter standard deviations over time.
        norm_sigmas_history: Dictionary of normalized sigma values over time.
        mean_target_history: Dictionary of mean target values over time (for multi-objective
            optimization). None if no targets were specified.
        sigma_target_history: Dictionary of target standard deviations over time.
        epochs_completed: Total number of epochs/generations completed.
        terminated_reason: String describing why optimization terminated.
        cmaes_sigma: Final step size of the CMA-ES algorithm.
        cmaes_C: Final covariance matrix of the CMA-ES algorithm.
    
    Example:
        >>> # Accessing optimization results
        >>> results = optimizer.optimize()
        >>> 
        >>> # Get best parameters and error
        >>> print(f"Best parameters: {results.best_parameters}")
        >>> print(f"Best fitness: {results.final_error}")
        >>> 
        >>> # Plot convergence history
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(12, 6))
        >>> plt.plot(results.mean_error_history)
        >>> plt.title(f"Optimization Convergence ({results.terminated_reason})")
        >>> plt.xlabel("Epoch")
        >>> plt.ylabel("Error")
        >>> plt.grid(True)
        >>> plt.show()
    """

    # Final parameters
    best_parameters: Dict[str, float]
    
    # Final error
    final_error: float
    
    # History data
    mean_error_history: List[float]
    sigma_error_history: List[float]
    mean_params_history: Dict[str, List[float]]
    sigma_params_history: Dict[str, List[float]]
    norm_sigmas_history: Dict[str, List[float]]
    
    # If targets were provided
    mean_target_history: Optional[Dict[str, List[float]]] = None
    sigma_target_history: Optional[Dict[str, List[float]]] = None
    
    # Metadata
    epochs_completed: int = None
    terminated_reason: str = None
    
    # CMAES specific data
    cmaes_sigma: float = None
    cmaes_C: Optional[np.ndarray] = None