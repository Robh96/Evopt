import numpy as np
import cloudpickle
from abc import ABC, abstractmethod
from contextlib import contextmanager
import os
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import concurrent.futures
from .directory_manager import DirectoryManager
from .utils import write_to_csv, format_array, extend_dict

class BaseOptimiser(ABC):
	"""
	Base class for optimisers, providing a common interface and functionality.
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
		max_workers=None,
	):
		"""
		Initialise the BaseOptimiser.

		Args:
			parameters (dict): A dictionary defining the parameters to optimise,
				where keys are parameter names and values are tuples of (min, max) bounds.
			evaluator: A callable that evaluates the parameters and returns an error value.
			batch_size (int): The number of solutions to evaluate in each epoch.
			directory_manager (DirectoryManager): An instance of the DirectoryManager
				to handle file and directory operations.
			sigma_threshold (float, optional): Threshold for sigma values to terminate optimisation. Defaults to 0.01.
			rand_seed (int, optional): Random seed for reproducibility. Defaults to 1.
			start_epoch (int, optional): Epoch to start from (for resuming). Defaults to None.
			verbose (bool, optional): Whether to print detailed information during optimisation. Defaults to False.
			n_epochs (int, optional): The number of epochs to run the optimisation for. If None, the optimisation runs until the termination criteria is met. Defaults to None.
			target_dict (dict, optional): If provided, the optimisation will be done to minimize the error between the target_dict and the output of the evaluator function. Defaults to None.
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
		self.max_workers = max_workers if max_workers else mp.cpu_count()
		self._file_lock = mp.Lock()  # For CSV file access synchronization
		
		self._executor = None
		if self.max_workers > 1:
			self._executor = concurrent.futures.ProcessPoolExecutor(
				max_workers=min(self.max_workers, mp.cpu_count())
		)

		self._mean_error_history = []
		self._sigma_error_history = []
		self._mean_params_history = {param: [] for param in self.parameters}
		self._sigma_params_history = {param: [] for param in self.parameters}
		self._norm_sigmas_history = {param: [] for param in self.parameters}
		self._mean_target_history = {target: [] for target in self.target_dict} if self.target_dict else None
		self._sigma_target_history = {target: [] for target in self.target_dict} if self.target_dict else None
		

	@property
	def get_init_sigmas(self):
		"""
		Calculate initial standard deviation values based on parameter bounds.

		Returns:
			np.ndarray: An array of initial standard deviation values for each parameter.
		"""
		return np.array([(max_val - min_val) / 4 for min_val, max_val in self.parameters.values()])

	@property
	def get_norm_bounds(self):
		"""
		Calculate bounds for the parameters normalised by the initial standard deviations.

		Returns:
			list: A list of tuples containing the normalised min and max bounds for each parameter.
		"""
		return [(min_val / std, max_val / std)
				for (min_val, max_val), std in zip(self.parameters.values(), self.init_sigmas)]

	@property
	def get_init_params(self):
		"""
		Generate initial parameters uniformly within the normalised bounds.

		Returns:
			list: A list of initial parameter values.
		"""
		# Generate initial parameters uniformly in the normalised bounds
		return [np.random.uniform(low, high) for low, high in self.norm_bounds]
	
	@property
	def mean_error(self):
		"""
		Get the historical mean errors.  This is a read-only property.
		"""
		return self._mean_error_history[:]

	@property
	def sigma_error(self):
		"""
		Get the historical sigma errors. This is a read-only property.
		"""
		return self._sigma_error_history[:]

	@property
	def mean_params(self):
		"""
		Get the historical mean parameters. This is a read-only property.
		"""
		return {p:v[:] for p,v in self._mean_params_history.items()}

	@property
	def sigma_params(self):
		"""
		Get the historical sigma parameters. This is a read-only property.
		"""
		return {p:v[:] for p,v in self._sigma_params_history.items()}

	@property
	def norm_sigmas(self):
		"""
		Get the historical normalised sigmas. This is a read-only property.
		"""
		return {p:v[:] for p,v in self._norm_sigmas_history.items()}

	@property
	def mean_targets(self):
		"""
		Get the historical mean parameters. This is a read-only property.
		"""
		return {p:v[:] for p,v in self._mean_target_history.items()}

	@property
	def sigma_targets(self):
		"""
		Get the historical mean parameters. This is a read-only property.
		"""
		return {p:v[:] for p,v in self._sigma_target_history.items()}

	def rescale_params(self, params):
		"""
		Rescale normalised parameters to their original scale.

		Args:
			params (np.ndarray): Normalised parameter values.

		Returns:
			np.ndarray: Rescaled parameter values.
		"""
		return params * self.init_sigmas


	def _write_result_to_csv(self, sol, error, param_dict, result_dict: dict = None):
		"""
		Write the results of a solution to a CSV file.
		The CSV file is structured as:
		 | Epoch | Solution | Error | Param1 | Param2 | ... | ParamN |

		Args:
			sol (int): The solution number.
			error (float): The error value for the solution.
			param_dict (dict): A dictionary of parameter values for the solution.
		"""
		result = {
			'epoch': self.current_epoch,
			'solution': sol,
			'error': error if error is not None else 'None',
			**(result_dict if result_dict is not None else {}),
			**param_dict
			}
		write_to_csv(result, self.dir_manager.results_csv, sort_columns=['epoch', 'solution'])

	def _write_epoch_to_csv(self, mean_error, sigma_error, mean_params, sigma_params, norm_sigmas, mean_targets=None, sigma_targets=None):
		"""
		Write epoch data to a CSV file.
		The CSV file is structured as:
		| epoch | mean Error | mean targetN | mean ParamN | sigma Error | sigma targetN | sigma ParamN | norm SigmaN |

		Args:
			mean_error (float): The mean error for the epoch.
			sigma_error (float): The standard deviation of the errors for the epoch.
			mean_params (np.ndarray): The mean parameter values for the epoch.
			sigma_params (np.ndarray): The standard deviation of the parameter values for the epoch.
			norm_sigmas (np.ndarray): The normalised sigma values for the parameters.
			mean_targets (list, optional): The mean target values for the epoch. Defaults to None.
			sigma_targets (list, optional): The standard deviation of the target values for the epoch. Defaults to None.
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

	def _update_history_and_log(self, mean_error, sigma_error, mean_params, sigma_params, norm_sigmas, mean_targets=None, sigma_targets=None):
		"""Update history arrays and write epoch data to log."""
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

	def print_solution(self, sol_id, params, error):
		"""
		Print the results of a solution to terminal and log.

		Args:
			sol_id (int): The solution ID.
			params (np.ndarray): The parameter values for the solution.
			error (float): The error value for the solution.
		"""
		if self.verbose:
			print(f"Epoch {self.current_epoch} | ({sol_id + 1}/{self.batch_size}) | Params: [{format_array(params)}] | Error: {'None' if error is None else f'{error:.3f}'}")

	def print_epoch(self, mean_error, sigma_error, mean_params, sigma_params, norm_sigmas):
		"""
		Print the epoch statistics to terminal and log

		Args:
			mean_error (float): The mean error for the epoch.
			sigma_error (float): The standard deviation of the errors for the epoch.
			mean_params (np.ndarray): The mean parameter values for the epoch.
			sigma_params (np.ndarray): The standard deviation of the parameter values for the epoch.
			norm_sigmas (np.ndarray): The normalised sigma values for the parameters.
		"""
		if self.verbose:
			print(f"Epoch {self.current_epoch} | Mean Error: {mean_error:.3f} | Sigma Error: {sigma_error:.3f}")
			print(f"Epoch {self.current_epoch} | Mean Parameters: [{format_array(mean_params)}] | Sigma parameters: [{format_array(sigma_params)}]")
			print(f"Epoch {self.current_epoch} | Normalised Sigma parameters: [{format_array(norm_sigmas)}]")

	@classmethod
	def working_directory(cls, path: str):
		"""
		A context manager which changes the working directory to the given
		path, and then changes it back to its previous value on exit.

		Args:
			path (str): The path to change the working directory to.
		"""
		@contextmanager
		def _change_directory_context(path):
			prev_cwd = os.getcwd()
			os.makedirs(path, exist_ok=True)
			os.chdir(path)
			try:
				yield
			finally:
				os.chdir(prev_cwd)
		return _change_directory_context(path)

	@classmethod
	def _evaluate_solution_worker(cls, args):
		"""
		Class method for evaluating a solution that can be pickled and sent to worker processes.
		
		Args:
			args (tuple): (
				sol_id (int),
				params (np.ndarray),
				param_names (list),
				current_epoch (int),
				solution_folder (str),
				evaluator_func (callable),
				target_dict (dict or None),
				dir_manager (DirectoryManager),
				verbose (bool)
			)
			
		Returns:
			tuple: (sol_id, error, result_dict, param_dict)
		"""
		# Unpack arguments
		(sol_id,
		params,
		param_names,
		solution_folder,
		pickled_evaluator,
		target_dict,
		verbose) = args
		
		# Unpickle the evaluator function
		try:
			evaluator_func = cloudpickle.loads(pickled_evaluator)
		except Exception as e:
			if verbose:
				print(f"Error unpickling evaluator for solution {sol_id}: {e}")

		# Convert parameters to dictionary
		param_dict = dict(zip(param_names, params))
		result_dict = None
		try:
			with cls.working_directory(solution_folder):
				error = evaluator_func(param_dict)

			# Process target dictionary if provided
			if target_dict and isinstance(error, dict):
				from .loss import calc_loss  # Import here to avoid circular imports
				loss = calc_loss(target_dict, error, verbose=False)
				result_dict = loss.observed_dict
				error = loss.combined_loss

			elif target_dict:
				if verbose:
					print(f"Error in solution {sol_id}: Expected dictionary, got {type(error)}")
				error = None

		except Exception as e:
			error = None
			if verbose:
				print(f"Error evaluating solution {sol_id}: {e}")

		# Clean up empty directory
		if os.path.exists(solution_folder) and len(os.listdir(solution_folder)) == 0:
			try:
				os.rmdir(solution_folder)
			except:
				pass  # Ignore errors during cleanup

		return sol_id, error, result_dict, param_dict

	def process_batch(self, solutions):
		"""
		Process a batch of solutions using concurrent.futures for parallelization.
		
		Args:
			solutions (list): A list of parameter value arrays.
			
		Returns:
			list: A list of error values for each solution.
		"""
		# Rescale solutions
		rescaled_solutions = [self.rescale_params(sol) for sol in solutions]
		
		# Pickle the evaluator function to handle complex cases (lambdas, notebook functions, etc.)
		pickled_evaluator = cloudpickle.dumps(self.evaluator)

		# Create arguments list for each solution with ALL necessary parameters
		solution_args = []
		for i, params in enumerate(rescaled_solutions):
			solution_folder = self.dir_manager.create_solution_folder(self.current_epoch, i)
			args = (
				i,                              # sol_id
				params,                         # params
				list(self.parameters.keys()),   # param_names
				solution_folder,                # solution_folder
				pickled_evaluator,              # pickled evaluator_func instead of direct reference
				self.target_dict,               # target_dict
				self.verbose                    # verbose
			)
			solution_args.append(args)
		
		# Process solutions in parallel or serial based on number of processes
		errors = []
		observed_dict = {}
		all_results = []
		
		# Use serial processing if num_processes is 1
		if self.num_processes <= 1:
			for args in solution_args:
				result = self._evaluate_solution_worker(args)
				all_results.append(result)
		else:
			try:
				# Submit tasks
				future_to_sol_id = {
					self._executor.submit(self._evaluate_solution_worker, args): args[0]
					for args in solution_args
				}
				for future in concurrent.futures.as_completed(future_to_sol_id):
					try:
						result = future.result(timeout=300)
						all_results.append(result)
					except concurrent.futures.TimeoutError:
						sol_id = future_to_sol_id[future]
						print(f"Solution {sol_id} timed out after 5 minutes")
					except Exception as exc:
						sol_id = future_to_sol_id[future]
						print(f"Solution {sol_id} generated an exception: {exc}")
						# Continue with other solutions
			except Exception as e:
				# Fall back to serial processing if parallel processing fails
				print(f"Warning: Parallel processing failed ({e}), falling back to serial processing")
				all_results = []
				for args in solution_args:
					result = self._evaluate_solution_worker(args)
					all_results.append(result)

		# Process all results in the main process
		with self._file_lock:
			for sol_id, error, result_dict, param_dict in all_results:
				self._write_result_to_csv(sol_id, error, param_dict, result_dict=result_dict)
				if error is not None:
					errors.append(error)
				if result_dict:
					extend_dict(observed_dict, result_dict)

		# remove solution folder dir if empty
		if len(os.listdir(os.path.dirname(solution_folder))) == 0:
			os.rmdir(os.path.dirname(solution_folder))

		# Calculate statistics and update history
		return self._process_batch_results(errors, rescaled_solutions, observed_dict)
	

	def _process_batch_results(self, errors, rescaled_solutions, observed_dict):
		"""
		Process the aggregated results from a batch of solutions.
		
		Args:
			errors (list): List of error values.
			rescaled_solutions (list): List of rescaled parameter arrays.
			observed_dict (dict): Dictionary of observed values.
			
		Returns:
			list: List of error values.
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
			mean_observed = {k: np.mean(v) for k, v in observed_dict.items()}
			sigma_observed = {k: np.std(v) for k, v in observed_dict.items()}
			
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
		"""Clean up resources when object is deleted."""
		if hasattr(self, '_executor') and self._executor:
			self._executor.shutdown(wait=False)
			self._executor = None
        
	def cleanup(self):
		"""Explicitly clean up resources."""
		if hasattr(self, '_executor') and self._executor:
			self._executor.shutdown(wait=False)
			self._executor = None

	def check_termination(self):
		"""
		Check if the termination criteria are met.
		The standard deviation of the parameter values
		is a proxy for the convergence of the optimisation.

		Returns:
			bool: True if all sigmas are below the threshold,
			or if the maximum number of epochs has been reached,
			False otherwise.
		"""
		pass

	@abstractmethod
	def setup_opt(self, epoch=None):
		"""
		Abstract method to set up the optimiser. Must be implemented by subclasses.

		Args:
			epoch (int, optional): The epoch number to start from. Defaults to None.
		"""
		pass

	@abstractmethod
	def optimise(self):
		"""
		Abstract method to run the optimisation. Must be implemented by subclasses.
		"""
		pass