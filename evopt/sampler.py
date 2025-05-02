import multiprocessing as mp
import numpy as np
from scipy.stats import qmc
import cloudpickle
import concurrent.futures
import os
import traceback
from .directory_manager import DirectoryManager
from .utils import ProcessPoolManager, working_directory, extend_dict, write_to_csv, format_array

class SobolSampler():
    def __init__(
            self,
            params: dict,
            evaluator: callable,
            n_samples: int = 32,
            rand_seed: int = None,
            target_dict: dict = None,
            max_workers: int = 1,
            cores_per_worker: int = 1,
            base_dir: str = None,
            dir_id: str = None,
            verbose: bool = False,
            directory_manager: DirectoryManager = None,
            **kwargs
            ):
        self.params = params
        self.evaluator = evaluator
        self.n_samples = n_samples
        self.rand_seed = rand_seed
        self.target_dict = target_dict
        self.max_workers = max_workers
        self.cores_per_worker = cores_per_worker
        self.base_dir = base_dir
        self.dir_id = dir_id
        self.verbose = verbose
        self.kwargs = kwargs
        self._file_lock = mp.Lock()  # For CSV file access synchronization
        np.random.seed(rand_seed)

        self.dir_manager = directory_manager
        
        self.process_manager = ProcessPoolManager(
            max_workers=max_workers, 
            cores_per_worker=cores_per_worker,
            **kwargs
        )
        
        self.executor = None if max_workers <= 1 else self.process_manager.initialize()

        self.completed_samples = self.dir_manager.load_sample_history()
        self.all_samples = None

    def sampler(self):
        l_bounds = [v[0] for v in self.params.values()]
        u_bounds = [v[1] for v in self.params.values()]
        
        if self.verbose:
            print(f"Generating {self.n_samples} Sobol samples...")
        sampler = qmc.Sobol(d=len(self.params), scramble=True, seed=self.rand_seed)
        unit_samples = sampler.random(self.n_samples)
        self.all_samples = qmc.scale(unit_samples, l_bounds, u_bounds)

        return self.all_samples
    
    def _write_result_to_csv(
                self,
                sample_idx: int,
                param_dict: dict,
                result_dict: dict = None,
                error: float = None
        ) -> None:
        """Write a sample's results to CSV file.
        
        Records the results of evaluating a single solution, including parameter values,
        error, and any additional result metrics. Results are appended to the CSV file
        managed by the DirectoryManager.

        Args:
            sample: Sample index within the study.
            param_dict: Dictionary of parameter values used for this solution.
            result_dict: Optional dictionary of additional metrics from the evaluation.
                Default is None.
            error: Optional error value for the solution (lower is better).
                Default is None.
                
        Note:
            This is an internal method called by process_samples().
        """

        result = {
            'sample': sample_idx,
            'error': error if error is not None else None,
            **param_dict,
            **({k: result_dict.get(k) for k in result_dict} if result_dict else {}),
        }
        
        write_to_csv(result, self.dir_manager.results_csv, sort_columns=['sample'])


    def print_solution(self, sample_id: int, params: np.ndarray, error: float = None) -> None:
        """Print information about a single evaluated sample.
        
        Displays details of a sample evaluation, including the solution ID,
        parameter values, and error (optional). Only prints if verbose mode is enabled.

        Args:
            sol_id: Solution identifier within the current epoch.
            params: Array of parameter values used for this solution.
            error: Error value for the solution (lower is better).
        """
        
        if self.verbose:
            print(f"Sample {sample_id} | ({sample_id + 1}/{self.n_samples}) | Params: [{format_array(params)}] | Error: {'None' if error is None else f'{error:.3f}'}")


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
                - sample_folder (str): Folder for solution-specific files
                - pickled_evaluator (bytes): Cloudpickled evaluator function
                - target_dict (dict): Dictionary of target values (optional)
                - verbose (bool): Whether to print detailed information

        Returns:
            tuple: (sol_id, error, result_dict, param_dict) where:
                - sol_id (int): Solution identifier
                - result_dict (dict): Additional metrics from evaluation (if any)
                - error (float): Error value or None if evaluation failed (if any)
                - param_dict (dict): Dictionary of parameter values
                
        Note:
            This is an internal method called by process_samples().
        """

        # Unpack arguments
        (sample_id,
        params,
        param_names,
        sample_folder,
        pickled_evaluator,
        target_dict,
        verbose) = args
        
        np.random.seed(1000 + sample_id)  # Deterministic unique seed per solution
        try:
            evaluator_func = cloudpickle.loads(pickled_evaluator)
        except Exception as e:
            print(f"Error unpickling evaluator for solution {sample_id}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

        # Convert parameters to dictionary
        param_dict = dict(zip(param_names, params))
        result_dict = None
        try:
            with working_directory(sample_folder):
                error = evaluator_func(param_dict)
                if isinstance(error, dict):
                    result_dict = error
                    error = None
            # Process target dictionary if provided
            if target_dict and isinstance(error, dict):
                from .loss import calc_loss  # Import here to avoid circular imports
                loss = calc_loss(target_dict, error, hard_to_soft_weight=0.9, method="mae")
                result_dict = loss.observed_dict
                error = loss.combined_loss

            elif target_dict:
                if verbose:
                    print(f"Error in solution {sample_id}: Expected dictionary, got {type(error)}")
                error = None

        except Exception as e:
            error = None
            #print(f"Error evaluating solution {sol_id}: {e}")
            #print(f"Traceback:\n{traceback.format_exc()}")
            # Clean up empty directory
            if os.path.exists(sample_folder) and len(os.listdir(sample_folder)) == 0:
                try:
                    os.rmdir(sample_folder)
                except:
                    pass  # Ignore errors during cleanup
            return sample_id, error, result_dict, param_dict

        # Clean up empty directory
        if os.path.exists(sample_folder) and len(os.listdir(sample_folder)) == 0:
            try:
                os.rmdir(sample_folder)
            except:
                pass  # Ignore errors during cleanup
        return sample_id, error, result_dict, param_dict



    def process_samples(self) -> list:
        """Process a batch of Sobol samples in parallel or serial mode.

        Handles the evaluation of multiple parameter sets (samples) using the
        provided evaluator function. Supports serial and parallel processing.

        Args:
            samples: List of scaled parameter value arrays (Sobol samples) to evaluate.

        Returns:
            list: A list containing the results for each sample. Each element is
                  a tuple: (sample_idx, error, result_dict, param_dict).
                  Returns None for samples that failed evaluation.

        Raises:
            Exception: If submitting tasks to the process pool fails catastrophically.
        """

        if self.all_samples is None:
            self.sampler()
        if self.all_samples is None or len(self.all_samples) != self.n_samples:
             raise RuntimeError("Sample generation failed or produced incorrect number of samples.")

        pickled_evaluator = cloudpickle.dumps(self.evaluator)
        sample_args = []

        param_names = list(self.params.keys())


        for idx, params_array in enumerate(self.all_samples):
            
            if idx in self.completed_samples:
                continue
            
            sample_folder = self.dir_manager.create_sample_folder(idx)
            
            args = (
                idx,                     # sample_idx
                params_array,            # params (already scaled)
                param_names,             # param_names
                sample_folder,           # sample_folder (shared directory)
                pickled_evaluator,       # pickled evaluator_function
                self.target_dict,        # target_dict (if any)
                self.verbose             # verbose
            )
            sample_args.append(args)

        # Initialise result containers
        num_to_process = len(sample_args)

        if num_to_process == 0:
            print("No new samples to process.")
            return []
        
        temp_result_dicts = [None] * num_to_process
        run_results = [None] * num_to_process
        
        def store_result(result, original_idx, run_idx):
            # result expected format: (sample_idx, error, result_dict, param_dict)
            if result is None: # Handle potential None return from worker on severe failure
                print(f"Warning: Evaluation worker returned None for sample {original_idx}")
                run_results[run_idx] = None # Store None to indicate failure
                return

            _idx, error, result_dict, param_dict = result
            if _idx != original_idx:
                print(f"Warning: Sample index mismatch! Expected {original_idx}, got {_idx}.")
                run_results[run_idx] = None
                return

            with self._file_lock:
                self._write_result_to_csv(_idx, param_dict, result_dict=result_dict, error=error)

            run_results[run_idx] = result
            
            if self.verbose:
                params_array = self.all_samples[original_idx]
                self.print_solution(_idx, params_array, error=error)

        if self.executor is None:
            # Serial processing
            print("Running evaluations in serial mode.")
            for run_idx, args in enumerate(sample_args):
                original_idx = args[0]
                try:
                    # Call the static worker method
                    result = self._evaluate_solution_worker(args)
                    store_result(result, original_idx, run_idx)
                except Exception as e:
                    print(f"Sample {original_idx} evaluation failed directly: {e}")
                    # Create a failure result tuple to store
                    params_dict = dict(zip(param_names, self.all_samples[original_idx]))
                    fail_result = (original_idx, None, None, params_dict)
                    store_result(fail_result, original_idx, run_idx) # Store failure information
                    continue

        else:
            try:
                futures = {self.executor.submit(self._evaluate_solution_worker, args): (args[0], run_idx)
                        for run_idx, args in enumerate(sample_args)}
                
            except Exception as e:
                print(f"Traceback:\n{traceback.format_exc()}")
                if self.executor._broken:
                    print("Process pool is broken - reinitializing")
                    self.process_manager.cleanup()
                    self.executor = self.process_manager.initialize()
                    return self.process_samples()

            for future in concurrent.futures.as_completed(futures):
                try:
                    original_idx, run_idx = futures[future]
                    result = future.result()  
                    store_result(result, original_idx, run_idx)

                except Exception as e:
                    params_dict = dict(zip(param_names, self.all_samples[original_idx]))
                    fail_result = (original_idx, None, None, params_dict)
                    store_result(fail_result, original_idx, run_idx) # Store failure information
                    continue
        
        return run_results

    def cleanup(self):
        """Clean up resources, particularly the process pool."""
        if hasattr(self, 'process_manager'):
            self.process_manager.cleanup()
            self.executor = None

    def __del__(self):
        """Attempt cleanup when the object is garbage collected."""
        self.cleanup()


def sample(
    params: dict,
    evaluator: callable,
    n_samples: int = 32,
    rand_seed: int = None,
    target_dict: dict = None,
    max_workers: int = 1,
    cores_per_worker: int = 1,
    base_dir: str = None,
    dir_id: str = None,
    verbose: bool = False,
    **kwargs
) -> list:
    """Sample Sobol sequences and evaluate them using the provided evaluator.

    Args:
        params (dict): Parameter space definition as a dictionary where keys are parameter names
            and values are tuples of (min_value, max_value) bounds.
        evaluator (callable): Function that evaluates a parameter set and returns either:
            - A single float value representing the error/fitness (lower is better)
            - A dictionary of observed values to be compared with target_dict
        n_samples (int): Number of Sobol samples to generate. Default is 32.
        rand_seed (int): Random seed for reproducibility. Default is None.
        target_dict (dict): Dictionary of target values for comparison. Default is None.
        max_workers (int): Maximum number of worker processes. Default is 1.
        cores_per_worker (int): Number of CPU cores per worker process. Default is 1.
        base_dir (str): Base directory for storing optimization results. If None, uses current directory.
        dir_id (str): Directory ID for organizing results. Default is None.
        verbose (bool): Whether to print detailed information during sampling and evaluation. Default is False.

    Returns:
        list: A list containing the results for each sample. Each element is
              a tuple: (sample_idx, error, result_dict, param_dict).
              Returns None for samples that failed evaluation.

    Raises:
        RuntimeError: If sample generation fails or produces incorrect number of samples.
    """
    
    directory_manager = DirectoryManager(
            base_dir = os.getcwd() if base_dir is None else base_dir,
            dir_id = dir_id
        )

    sampler = SobolSampler(
        params=params,
        evaluator=evaluator,
        n_samples=n_samples,
        rand_seed=rand_seed,
        target_dict=target_dict,
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
        base_dir=base_dir,
        dir_id=dir_id,
        verbose=verbose,
        directory_manager=directory_manager,
        **kwargs
    )
    
    try:
        with directory_manager.logger:
            results = sampler.process_samples()
        return results
    finally:
        if hasattr(sampler, 'cleanup'):
            sampler.cleanup() # closes process workers