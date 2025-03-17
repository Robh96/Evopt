"""black-box optimization base implementation.

This module provides an abstract base class for implementing black-box optimization
algorithms. It handles common functionality such as parameter handling, parallel evaluation
of solutions, logging, and statistics calculation.

The BaseOptimizer class is designed to be extended by specific black-box optimization algorithm
implementations like CMA-ES, DE, PSO, etc. At the current stage only CMA-ES is implemented.

Example:
    Creating a custom optimizer by extending BaseOptimizer:
    
    >>> class MyOptimizer(BaseOptimizer):
    ...     def setup_opt(self, epoch=None):
    ...         # Initialize optimization algorithm
    ...         pass
    ...         
    ...     def optimize(self):
    ...         # Run optimization algorithm
    ...         while not self.check_termination():
    ...             solutions = self.generate_solutions()
    ...             errors = self.process_batch(solutions)
    ...             self.update_algorithm(solutions, errors)
    ...         return self.best_solution
    ...
    ...     def check_termination(self):
    ...         # Check if optimization should stop
    ...         return self.current_epoch >= self.n_epochs
"""

import numpy as np
import cloudpickle
from abc import ABC, abstractmethod
import os
import traceback
import concurrent.futures
import multiprocessing as mp
from .directory_manager import DirectoryManager
from .utils import write_to_csv, format_array, extend_dict, ProcessPoolManager, working_directory

class BaseOptimizer(ABC):
    """Abstract base class for evolutionary optimization algorithms.
    
    This class provides the foundation for implementing evolutionary optimization algorithms,
    handling common tasks such as parameter management, parallel solution evaluation,
    statistics tracking, and result logging. It abstracts away the infrastructure details,
    allowing subclasses to focus on algorithm-specific implementation.
    
    The class maintains a history of optimization metrics across epochs, supports both
    serial and parallel evaluation of solutions, and provides detailed logging of
    optimization progress.
    
    Attributes:
        parameters (dict): Parameter definitions with (min, max) bounds.
        target_dict (dict, optional): Target values to optimize towards.
        evaluator (callable): Function that evaluates parameter sets.
        n_epochs (int, optional): Maximum number of epochs to run.
        batch_size (int): Number of solutions to evaluate per epoch.
        dir_manager (DirectoryManager): Manages output directories and files.
        sigma_threshold (float): Convergence threshold for normalized sigmas.
        rand_seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print detailed progress information.
        current_epoch (int): The current optimization epoch.
        max_workers (int): Maximum number of parallel workers.
        
    Note:
        This is an abstract class and cannot be instantiated directly.
        Subclasses must implement the `setup_opt`, `optimize`, and `check_termination` methods.
    """

    def __init__(
        self,
        parameters: dict,
        evaluator,
        batch_size: int,
        directory_manager: DirectoryManager,
        sigma_threshold: float = 0.1,
        rand_seed: int = 1,
        start_epoch: int = None,
        verbose: bool = True,
        n_epochs: int = None,
        target_dict: dict = None,
        max_workers: int = 1,
        cores_per_worker: int = 1,
        **kwargs
    ):
        """Initialize the BaseOptimizer.

        Args:
            parameters: Dictionary of parameters to optimize. Keys are parameter names,
                values are tuples of (min, max) bounds.
            evaluator: Callable that takes a dictionary of parameter values and returns
                either an error value (float) or a dictionary of observed values.
            batch_size: Number of solutions to evaluate in each epoch.
            directory_manager: DirectoryManager instance to handle file I/O.
            sigma_threshold: Convergence threshold for normalized sigma values.
                Default is 0.1.
            rand_seed: Seed for random number generation to ensure reproducibility.
                Default is 1.
            start_epoch: Starting epoch number, useful for resuming optimization.
                Default is None (start from 0).
            verbose: Whether to print detailed progress information.
                Default is True.
            n_epochs: Maximum number of epochs to run. If None, runs until convergence.
                Default is None.
            target_dict: Dictionary of target values to optimize towards. Keys should match
                the keys returned by the evaluator. Default is None.
            max_workers: Maximum number of concurrent workers for parallel evaluation.
                Default is 1 (serial processing).
            cores_per_worker: CPU cores to allocate per worker.
                Default is 1.
                
        Example:
            >>> params = {'x': (0, 10), 'y': (-5, 5)}
            >>> def evaluator(params):
            ...     return (params['x'] - 5)**2 + params['y']**2
            >>> dir_manager = DirectoryManager('./optimization_results')
            >>> optimizer = MyOptimizer(
            ...     parameters=params,
            ...     evaluator=evaluator,
            ...     batch_size=10,
            ...     directory_manager=dir_manager,
            ...     n_epochs=50,
            ...     max_workers=4
            ... )
        """

        self.parameters = parameters
        self.target_dict = target_dict
        self.evaluator = evaluator
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dir_manager = directory_manager
        self.sigma_threshold = sigma_threshold
        self.rand_seed = rand_seed
        self.start_epoch = start_epoch
        self.verbose = verbose
        self.current_epoch = 0 # Previously None
        np.random.seed(rand_seed)
        self.init_sigmas = self.get_init_sigmas
        self.norm_bounds = self.get_norm_bounds
        self.init_params = self.get_init_params
        self.max_workers = max_workers
        self._file_lock = mp.Lock()  # For CSV file access synchronization

        self._mean_error_history = []
        self._sigma_error_history = []
        self._mean_params_history = {param: [] for param in self.parameters}
        self._sigma_params_history = {param: [] for param in self.parameters}
        self._norm_sigmas_history = {param: [] for param in self.parameters}
        self._mean_target_history = {target: [] for target in self.target_dict} if self.target_dict else None
        self._sigma_target_history = {target: [] for target in self.target_dict} if self.target_dict else None
        
        self.process_manager = ProcessPoolManager(
            max_workers=max_workers, 
            cores_per_worker=cores_per_worker
        )
        self.executor = None if max_workers <= 1 else self.process_manager.initialize()

    @property
    def get_init_sigmas(self) -> np.ndarray:
        """Calculate initial standard deviations based on parameter bounds.
        
        Computes initial sigma values as 1/4 of the range between min and max
        for each parameter, providing a reasonable starting point for exploration.
        
        Returns:
            np.ndarray: Array of initial standard deviation values for each parameter.
            
        Example:
            >>> optimizer = MyOptimizer(parameters={'x': (0, 10), 'y': (-5, 5)}, ...)
            >>> optimizer.get_init_sigmas
            array([2.5, 2.5])  # 1/4 of parameter ranges
        """

        return np.array([(max_val - min_val) / 4 for min_val, max_val in self.parameters.values()])

    @property
    def get_norm_bounds(self) -> list:
        """Calculate normalized parameter bounds.
        
        Normalizes the parameter bounds by dividing by the initial sigma values.
        This creates a unified scale for parameters with different ranges.
        
        Returns:
            list: List of (normalized_min, normalized_max) tuples for each parameter.
            
        Example:
            >>> optimizer = MyOptimizer(parameters={'x': (0, 10), 'y': (-5, 5)}, ...)
            >>> optimizer.get_norm_bounds
            [(0.0, 4.0), (-2.0, 2.0)]  # Bounds divided by sigmas
        """

        return [(min_val / std, max_val / std)
                for (min_val, max_val), std in zip(self.parameters.values(), self.init_sigmas)]

    @property
    def get_init_params(self) -> list:
        """Generate initial parameters within normalized bounds.
        
        Creates random initial parameter values uniformly distributed within
        the normalized parameter bounds.
        
        Returns:
            list: List of initial normalized parameter values.
            
        Example:
            >>> np.random.seed(1)  # For reproducible example
            >>> optimizer = MyOptimizer(parameters={'x': (0, 10), 'y': (-5, 5)}, ...)
            >>> optimizer.get_init_params
            [2.17, -0.45]  # Random values within normalized bounds
        """

        # Generate initial parameters uniformly in the normalised bounds
        return [np.random.uniform(low, high) for low, high in self.norm_bounds]
    
    @property
    def mean_error(self) -> list:
        """Get the historical mean error values.
        
        Returns:
            list: Copy of the mean error history list.
            
        Example:
            >>> optimizer.mean_error
            [10.5, 8.2, 6.7, 4.3, 2.1]  # Error trajectory across epochs
        """

        return self._mean_error_history[:]

    @property
    def sigma_error(self) -> list:
        """Get the historical error standard deviations.
        
        Returns:
            list: Copy of the error standard deviation history list.
            
        Example:
            >>> optimizer.sigma_error
            [5.2, 4.3, 3.1, 2.5, 1.2]  # Error variance trajectory
        """

        return self._sigma_error_history[:]

    @property
    def mean_params(self) -> dict:
        """Get the historical mean parameter values.
        
        Returns:
            dict: Dictionary with parameter names as keys and lists of historical
                mean values as values. Each value is a copy of the internal list.
                
        Example:
            >>> optimizer.mean_params
            {'x': [5.2, 5.1, 5.05, 5.01], 'y': [0.5, 0.3, 0.1, 0.05]}
        """

        return {p:v[:] for p,v in self._mean_params_history.items()}

    @property
    def sigma_params(self) -> dict:
        """Get the historical parameter standard deviations.
        
        Returns:
            dict: Dictionary with parameter names as keys and lists of historical
                standard deviation values as values. Each value is a copy of the internal list.
                
        Example:
            >>> optimizer.sigma_params
            {'x': [2.0, 1.5, 1.0, 0.5], 'y': [1.0, 0.7, 0.4, 0.2]}
        """

        return {p:v[:] for p,v in self._sigma_params_history.items()}

    @property
    def norm_sigmas(self) -> dict:
        """Get the historical normalized sigma values.
        
        Normalized sigmas represent the standard deviation divided by initial sigma,
        serving as a measure of convergence.
        
        Returns:
            dict: Dictionary with parameter names as keys and lists of historical
                normalized sigma values as values. Each value is a copy of the internal list.
                
        Example:
            >>> optimizer.norm_sigmas
            {'x': [1.0, 0.7, 0.5, 0.2], 'y': [1.0, 0.8, 0.5, 0.3]}
        """

        return {p:v[:] for p,v in self._norm_sigmas_history.items()}

    @property
    def mean_targets(self) -> dict:
        """Get the historical mean target values.
        
        Returns:
            dict: Dictionary with target names as keys and lists of historical
                mean observed values as values. Each value is a copy of the internal list.
                
        Example:
            >>> optimizer.mean_targets
            {'stress': [250, 240, 230, 225], 'weight': [120, 118, 117, 116.5]}
        """

        return {p:v[:] for p,v in self._mean_target_history.items()}

    @property
    def sigma_targets(self) -> dict:
        """Get the historical target standard deviations.
        
        Returns:
            dict: Dictionary with target names as keys and lists of historical
                standard deviation values as values. Each value is a copy of the internal list.
                
        Example:
            >>> optimizer.sigma_targets
            {'stress': [25, 20, 15, 10], 'weight': [8, 6, 4, 3]}
        """

        return {p:v[:] for p,v in self._sigma_target_history.items()}

    def rescale_params(self, params: np.ndarray) -> np.ndarray:
        """Rescale normalized parameters to their original scale.
        
        Converts normalized parameter values (used internally by optimization algorithms)
        back to their original scale by multiplying by the initial sigma values.

        Args:
            params: Normalized parameter values.

        Returns:
            np.ndarray: Parameter values in their original scale.
            
        Example:
            >>> normalized_params = np.array([1.0, -0.5])
            >>> optimizer.rescale_params(normalized_params)
            array([2.5, -1.25])  # Assuming init_sigmas = [2.5, 2.5]
        """

        return params * self.init_sigmas


    def _write_result_to_csv(
                self,
                sol: int,
                error: float,
                param_dict: dict,
                result_dict: dict = None
        ) -> None:
        """Write a solution's results to CSV file.
        
        Records the results of evaluating a single solution, including parameter values,
        error, and any additional result metrics. Results are appended to the CSV file
        managed by the DirectoryManager.

        Args:
            sol: Solution number within the current epoch.
            error: Error value for the solution (lower is better).
            param_dict: Dictionary of parameter values used for this solution.
            result_dict: Optional dictionary of additional metrics from the evaluation.
                Default is None.
                
        Note:
            This is an internal method called by process_batch().
        """

        result = {
            'epoch': self.current_epoch,
            'solution': sol,
            'error': error if error is not None else None,
            **({k: result_dict.get(k) for k in self.target_dict if k in result_dict} if result_dict else {}),
            **param_dict
        }
        
            
        write_to_csv(result, self.dir_manager.results_csv, sort_columns=['epoch', 'solution'])

    def _write_epoch_to_csv(
        self, 
        mean_error: float, 
        sigma_error: float, 
        mean_params: np.ndarray, 
        sigma_params: np.ndarray,
        norm_sigmas: np.ndarray,
        mean_targets=None,
        sigma_targets=None
    ) -> None:
        """Write epoch statistics to CSV file.
        
        Records aggregated statistics for the current epoch, including mean and standard
        deviation of errors, parameters, and target values (if applicable).

        Args:
            mean_error: Mean error across all solutions in the epoch.
            sigma_error: Standard deviation of errors in the epoch.
            mean_params: Array of mean parameter values.
            sigma_params: Array of parameter standard deviations.
            norm_sigmas: Array of normalized sigma values.
            mean_targets: Dictionary of mean target values. Default is None.
            sigma_targets: Dictionary of target standard deviations. Default is None.
            
        Note:
            This is an internal method called by _update_history_and_log().
        """

        epoch_data = {
            'epoch': self.current_epoch,
            'mean error': mean_error,
            **({f"mean {target}": mean for target, mean in zip(self.target_dict.keys(), mean_targets)} if mean_targets else {}),
            **{f"mean {param}": mean for param, mean in zip(self.parameters.keys(), mean_params)},
            'sigma error': sigma_error,
            **({f"sigma {target}": sigma for target, sigma in zip(self.target_dict.keys(), sigma_targets)} if sigma_targets else {}),
            **{f"sigma {param}": sigma for param, sigma in zip(self.parameters.keys(), sigma_params)},
            **{f"norm sigma {param}": norm_sigma for param, norm_sigma in zip(self.parameters.keys(), norm_sigmas)}
        }
        write_to_csv(epoch_data, self.dir_manager.epochs_csv, sort_columns=['epoch'])

    def _update_history_and_log(
        self,
        mean_error: float,
        sigma_error: float, 
        mean_params: np.ndarray,
        sigma_params: np.ndarray,
        norm_sigmas: np.ndarray,
        mean_targets=None,
        sigma_targets=None
    ) -> None:
        """Update optimization history and log epoch statistics.
        
        Updates internal history arrays with the current epoch's statistics and
        writes the data to the epoch CSV file. Also prints epoch statistics if
        verbose mode is enabled.

        Args:
            mean_error: Mean error across all solutions in the epoch.
            sigma_error: Standard deviation of errors in the epoch.
            mean_params: Array of mean parameter values.
            sigma_params: Array of parameter standard deviations.
            norm_sigmas: Array of normalized sigma values.
            mean_targets: Dictionary of mean target values. Default is None.
            sigma_targets: Dictionary of target standard deviations. Default is None.
            
        Note:
            This is an internal method called by _process_batch_results().
        """

        # Print epoch statistics
        self.print_epoch(mean_error, sigma_error, mean_params, sigma_params, norm_sigmas)
        
        # Write epoch data to CSV
        self._write_epoch_to_csv(mean_error, sigma_error, mean_params, sigma_params, norm_sigmas,
                                mean_targets=mean_targets, sigma_targets=sigma_targets)
        
        # Update history
        self._mean_error_history.append(mean_error)
        self._sigma_error_history.append(sigma_error)
        
        # Update parameter history
        for i, p in enumerate(self.parameters):
            self._mean_params_history[p].append(mean_params[i])
            self._sigma_params_history[p].append(sigma_params[i])
            self._norm_sigmas_history[p].append(norm_sigmas[i])
        
        # Update target history if available
        if self.target_dict and mean_targets and sigma_targets:
            for i, t in enumerate(self.target_dict):
                self._mean_target_history[t].append(list(mean_targets)[i])
                self._sigma_target_history[t].append(list(sigma_targets)[i])

    def print_solution(self, sol_id: int, params: np.ndarray, error: float) -> None:
        """Print information about a single evaluated solution.
        
        Displays details of a solution evaluation, including the solution ID,
        parameter values, and error. Only prints if verbose mode is enabled.

        Args:
            sol_id: Solution identifier within the current epoch.
            params: Array of parameter values used for this solution.
            error: Error value for the solution (lower is better).
        """
        
        if self.verbose:
            print(f"Epoch {self.current_epoch} | ({sol_id + 1}/{self.batch_size}) | Params: [{format_array(params)}] | Error: {'None' if error is None else f'{error:.3f}'}")

    def print_epoch(
        self,
        mean_error: float,
        sigma_error: float,
        mean_params: np.ndarray,
        sigma_params: np.ndarray,
        norm_sigmas: np.ndarray
    ) -> None:
        """Print statistics for the current epoch.
        
        Displays aggregated statistics for the current epoch, including mean error,
        error standard deviation, and parameter statistics. Only prints if verbose 
        mode is enabled.

        Args:
            mean_error: Mean error across all solutions in the epoch.
            sigma_error: Standard deviation of errors in the epoch.
            mean_params: Array of mean parameter values.
            sigma_params: Array of parameter standard deviations.
            norm_sigmas: Array of normalized sigma values.
        """

        if self.verbose:
            print(f"Epoch {self.current_epoch} | Mean Error: {mean_error:.3f} | Sigma Error: {sigma_error:.3f}")
            print(f"Epoch {self.current_epoch} | Mean Parameters: [{format_array(mean_params)}] | Sigma parameters: [{format_array(sigma_params)}]")
            print(f"Epoch {self.current_epoch} | Normalised Sigma parameters: [{format_array(norm_sigmas)}]")

    @classmethod
    def _evaluate_solution_worker(cls, args: tuple) -> tuple:
        """Evaluate a single solution in a worker process.
        
        Static method that evaluates a solution with the provided parameters.
        Designed to be used with process pools for parallel evaluation.

        Args:
            args: Tuple containing:
                - sol_id (int): Solution identifier
                - params (np.ndarray): Parameter values to evaluate
                - param_names (list): List of parameter names
                - solution_folder (str): Folder for solution-specific files
                - pickled_evaluator (bytes): Cloudpickled evaluator function
                - target_dict (dict): Dictionary of target values (optional)
                - verbose (bool): Whether to print detailed information

        Returns:
            tuple: (sol_id, error, result_dict, param_dict) where:
                - sol_id (int): Solution identifier
                - error (float): Error value or None if evaluation failed
                - result_dict (dict): Additional metrics from evaluation (if any)
                - param_dict (dict): Dictionary of parameter values
                
        Note:
            This is an internal method called by process_batch().
        """

        # Unpack arguments
        (sol_id,
        params,
        param_names,
        solution_folder,
        pickled_evaluator,
        target_dict,
        verbose) = args
        
        np.random.seed(1000 + sol_id)  # Deterministic unique seed per solution
        try:
            evaluator_func = cloudpickle.loads(pickled_evaluator)
        except Exception as e:
            print(f"Error unpickling evaluator for solution {sol_id}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

        # Convert parameters to dictionary
        param_dict = dict(zip(param_names, params))
        result_dict = None
        try:
            with working_directory(solution_folder):
                error = evaluator_func(param_dict)

            # Process target dictionary if provided
            if target_dict and isinstance(error, dict):
                from .loss import calc_loss  # Import here to avoid circular imports
                loss = calc_loss(target_dict, error, hard_to_soft_weight=0.95, method="mae")
                result_dict = loss.observed_dict
                error = loss.combined_loss

            elif target_dict:
                if verbose:
                    print(f"Error in solution {sol_id}: Expected dictionary, got {type(error)}")
                error = None

        except Exception as e:
            error = None
            print(f"Error evaluating solution {sol_id}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return sol_id, error, result_dict, param_dict

        # Clean up empty directory
        if os.path.exists(solution_folder) and len(os.listdir(solution_folder)) == 0:
            try:
                os.rmdir(solution_folder)
            except:
                pass  # Ignore errors during cleanup
        return sol_id, error, result_dict, param_dict

    def process_batch(self, solutions: list) -> list:
        """Process a batch of solutions in parallel or serial mode.
        
        This method handles the evaluation of multiple parameter sets (solutions)
        using the provided evaluator function. It supports both serial and parallel
        processing based on max_workers setting.

        Args:
            solutions: List of parameter value arrays to evaluate.

        Returns:
            list: List of error values for each solution.
            
        Raises:
            Exception: If solution evaluation fails.
            
        Example:
            >>> # Generate solutions with some algorithm
            >>> solutions = [[1.2, 3.4], [2.3, 1.4], [3.1, 2.8]]
            >>> errors = optimizer.process_batch(solutions)
            >>> print(errors)
            [12.5, 8.7, 15.2]  # Error value for each solution
        """

        # Rescale solutions
        rescaled_solutions = [self.rescale_params(sol) for sol in solutions]
        pickled_evaluator = cloudpickle.dumps(self.evaluator)
        solution_args = []

        for idx, params in enumerate(rescaled_solutions):
            solution_folder = self.dir_manager.create_solution_folder(self.current_epoch, idx)
            args = (
                idx,                            # sol_id
                params,                         # params
                list(self.parameters.keys()),   # param_names
                solution_folder,                # solution_folder
                pickled_evaluator,              # pickled evaluator_func instead of direct reference
                self.target_dict,               # target_dict
                self.verbose                    # verbose
            )
            solution_args.append(args)
        
        # Initialise result containers
        all_results = [None] * len(solution_args)
        temp_result_dicts = [None] * len(solution_args) 
        errors = [None] * len(solution_args)
        
        def store_result(result, sol_idx):
            sol_id, error, result_dict, param_dict = result
            with self._file_lock:
                self._write_result_to_csv(sol_id, error, param_dict, result_dict=result_dict)
            
            # Store in correct position for later processing
            all_results[sol_idx] = result
            errors[sol_idx] = error
            temp_result_dicts[sol_idx] = result_dict
            
            if self.verbose:
                self.print_solution(sol_id, rescaled_solutions[sol_id], error)
        
        if self.executor is None:
            # Serial processing
            for args in solution_args:
                idx = args[0]
                try:
                    result = self._evaluate_solution_worker(args)
                    store_result(result, idx)

                except Exception as e:
                    print(f"Solution {idx} failed with error: {e}")
                    print(f"Traceback:\n{traceback.format_exc()}")
                    result_dict = {k: None for k in self.target_dict} if self.target_dict else None
                    result = (idx, None, result_dict, dict(zip(self.parameters.keys(), rescaled_solutions[idx])))
                    store_result(result, idx)
                    continue

        else:
            try:
                futures = {self.executor.submit(self._evaluate_solution_worker, args): args[0]
                        for args in solution_args}
            except Exception as e:
                print(f"Traceback:\n{traceback.format_exc()}")
                if self.executor._broken:
                    print("Process pool is broken - reinitializing")
                    self.process_manager.cleanup()
                    self.executor = self.process_manager.initialize()
                    return self.process_batch(solutions)

            for future in concurrent.futures.as_completed(futures):
                try:
                    idx = futures[future]
                    result = future.result()  
                    store_result(result, idx)

                except Exception as e:
                    print(f"Solution {solution_args[idx][0]} failed: {e}")
                    print(f"Traceback:\n{traceback.format_exc()}")
                    result_dict = {k: None for k in self.target_dict} if self.target_dict else None
                    result = (solution_args[idx][0], None, result_dict, 
                            dict(zip(self.parameters.keys(), rescaled_solutions[idx])))
                    store_result(result, idx)
                    continue
                        
        observed_dict = {}
        for result_dict in temp_result_dicts:
            if result_dict:
                extend_dict(observed_dict, result_dict)	

        try:
            if len(os.listdir(os.path.dirname(solution_folder))) == 0:
                os.rmdir(os.path.dirname(solution_folder))
        except Exception:
            pass

        return self._process_batch_results(errors, rescaled_solutions, observed_dict)
    

    def _process_batch_results(
        self, 
        errors: list, 
        rescaled_solutions: list,
        observed_dict: dict
    ) -> list:
        """Process results from a batch of solution evaluations.
        
        Aggregates and analyzes the results from evaluating multiple solutions,
        calculates statistics, and updates the optimization history.

        Args:
            errors: List of error values from each solution.
            rescaled_solutions: List of parameter arrays in their original scale.
            observed_dict: Dictionary of observed values from the evaluations.

        Returns:
            list: List of error values (with None values replaced by the mean).
            
        Raises:
            ValueError: If all errors are None, indicating all evaluations failed.
            
        Note:
            This is an internal method called by process_batch().
        """

        # Handle case where all errors are None
        valid_err = [err for err in errors if err is not None]
        if not valid_err:
            raise ValueError("All errors are None")
        
        # Replace None errors with mean of valid errors
        errors = [err if err is not None else float(np.mean(valid_err)) for err in errors]
        
        # Calculate statistics
        mean_error = np.mean(errors)
        sigma_error = np.std(errors)
        mean_params = np.mean(rescaled_solutions, axis=0)
        sigma_params = np.std(rescaled_solutions, axis=0)
        norm_sigmas = sigma_params / self.init_sigmas
        
        # Process target observations if available
        if self.target_dict:
            mean_observed = {
                k: np.mean([v for v in observed_dict.get(k) if v is not None]) 
                for k in self.target_dict
            }
            sigma_observed = {
                k: np.std([v for v in observed_dict.get(k) if v is not None]) 
                for k in self.target_dict
            }
            
            # Update history and write to CSV
            self._update_history_and_log(
                mean_error, sigma_error, mean_params, sigma_params, norm_sigmas,
                mean_targets=mean_observed.values(), 
                sigma_targets=sigma_observed.values()
            )
        else:
            # Update history and write to CSV without targets
            self._update_history_and_log(
                mean_error, sigma_error, mean_params, sigma_params, norm_sigmas
            )
        
        return errors

    def __del__(self):
        """Clean up resources when the optimizer is garbage collected.
        
        Ensures that all resources used by the optimizer, particularly process pools,
        are properly released even if cleanup() is not explicitly called.
        """

        self.cleanup()
        
    def cleanup(self):
        """Explicitly clean up resources used by the optimizer.
        
        Releases all resources used by the optimizer, including any process pools.
        This method should be called when the optimizer is no longer needed.
        
        Example:
            >>> # After optimization is complete
            >>> optimizer.cleanup()
        """

        if hasattr(self, 'process_manager'):
            self.process_manager.cleanup()

    @abstractmethod
    def check_termination(self) -> bool:
        """Check if optimization should terminate.
        
        Determines whether the optimization process should stop based on
        convergence criteria or maximum epochs.
        
        Returns:
            bool: True if optimization should stop, False otherwise.
            
        Note:
            This is an abstract method that must be implemented by subclasses.
            
        Example:
            >>> class MyOptimizer(BaseOptimizer):
            ...     def check_termination(self):
            ...         if self.current_epoch >= self.n_epochs:
            ...             return True
            ...         return all(ns < self.sigma_threshold 
            ...                   for ns in self.norm_sigmas.values())
        """

        pass

    @abstractmethod
    def setup_opt(self, epoch: int = None):
        """Set up the optimization algorithm.
        
        Initializes or reinitializes the optimization algorithm, potentially
        starting from a specific epoch (for resuming interrupted optimizations).
        
        Args:
            epoch: Starting epoch number. If None, starts from 0.
            
        Note:
            This is an abstract method that must be implemented by subclasses.
            
        Example:
            >>> class MyCmaesOptimizer(BaseOptimizer):
            ...     def setup_opt(self, epoch=None):
            ...         self.es = cma.CMAEvolutionStrategy(
            ...             self.init_params,
            ...             0.5,
            ...             {'seed': self.rand_seed}
            ...         )
            ...         if epoch is not None:
            ...             # Load state from checkpoint if available
            ...             self.es = self.dir_manager.load_checkpoint(epoch)
            ...             self.current_epoch = epoch
        """

        pass

    @abstractmethod
    def optimize(self):
        """Run the optimization process.
        
        Executes the optimization algorithm until termination criteria are met.
        
        Returns:
            A representation of the best solution found.
            
        Note:
            This is an abstract method that must be implemented by subclasses.
            
        Example:
            >>> class MyCmaesOptimizer(BaseOptimizer):
            ...     def optimize(self):
            ...         self.setup_opt(epoch=self.start_epoch)
            ...         while not self.check_termination():
            ...             solutions = self.es.ask()
            ...             errors = self.process_batch(solutions)
            ...             self.es.tell(solutions, errors)
            ...             self.current_epoch += 1
            ...         return {p: v[-1] for p, v in self.mean_params.items()}
        """
        
        pass