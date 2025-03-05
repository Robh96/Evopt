import numpy as np
from abc import ABC, abstractmethod
import os
from .directory_manager import DirectoryManager
from .utils import write_to_csv, format_array, extend_dict
from .loss import calc_loss

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
		write_to_csv(result, self.dir_manager.results_csv)

	def _write_epoch_to_csv(self, mean_error, sigma_error, mean_params, sigma_params, norm_sigmas, mean_targets=None, sigma_targets=None):
		"""
		Write epoch data to a CSV file.
		The CSV file is structured as:
		 | Epoch | Mean Error | Mean ParamN | Sigma Error | Sigma ParamN | Norm SigmaN |

		Args:
			mean_error (float): The mean error for the epoch.
			sigma_error (float): The standard deviation of the errors for the epoch.
			mean_params (np.ndarray): The mean parameter values for the epoch.
			sigma_params (np.ndarray): The standard deviation of the parameter values for the epoch.
			norm_sigmas (np.ndarray): The normalised sigma values for the parameters.
		"""
		epoch_data = {
			'Epoch': self.current_epoch,
			'Mean error': mean_error,
			**({f"Mean {target}": mean for target, mean in zip(self.target_dict.keys(), mean_targets)} if mean_targets else {}),
			**{f"Mean {param}": mean for param, mean in zip(self.parameters.keys(), mean_params)},
			'Sigma error': sigma_error,
			**({f"Sigma {target}": sigma for target, sigma in zip(self.target_dict.keys(), sigma_targets)} if sigma_targets else {}),
			**{f"Sigma {param}": sigma for param, sigma in zip(self.parameters.keys(), sigma_params)},
			**{f"Norm sigma {param}": norm_sigma for param, norm_sigma in zip(self.parameters.keys(), norm_sigmas)}
		}
		write_to_csv(epoch_data, self.dir_manager.epochs_csv)

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

	def process_batch(self, solutions):
		"""
		Process a batch of solutions, evaluate them, and write the results to CSV.

		Args:
			solutions (list): A list of parameter value arrays.

		Returns:
			list: A list of error values for each solution.
		"""
		# in series, future work to parallelise
		rescaled_solutions = [self.rescale_params(sol) for sol in solutions]
		observed_dict = {}
		errors = []
		for sol, params in enumerate(rescaled_solutions):
			param_dict = dict(zip(self.parameters.keys(), params))
			solution_folder = self.dir_manager.create_solution_folder(self.current_epoch, sol)
			with self.dir_manager.working_directory(solution_folder):
				error = self.evaluator(param_dict)
			if self.target_dict:
				if isinstance(error, dict):
					loss = calc_loss(self.target_dict, error, self.verbose)
					error = loss.combined_loss
					new_dict = loss.observed_dict
					extend_dict(observed_dict, new_dict)
				else:
					raise ValueError("Target dict provided but evaluator output is not a dictionary.")
			# if solution directory is empty, delete it
			if len(os.listdir(solution_folder)) == 0:
				os.rmdir(solution_folder)

			self.print_solution(sol, params, error)
			self._write_result_to_csv(sol, error, param_dict, result_dict=new_dict if self.target_dict else None)
			errors.append(error)
			
		valid_err = [err for err in errors if err is not None]
		if not valid_err:
			raise ValueError("All errors are None")
		errors = [err if err is not None else float(np.mean(valid_err)) for err in errors]

		mean_error = np.mean(errors)
		sigma_error = np.std(errors)
		mean_params = np.mean(rescaled_solutions, axis=0)
		sigma_params = np.std(rescaled_solutions, axis=0)
		norm_sigmas = sigma_params / self.init_sigmas
		mean_observed = {k: np.mean(v) for k, v in observed_dict.items() if self.target_dict}
		sigma_observed = {k: np.std(v) for k, v in observed_dict.items() if self.target_dict}
		
		self.print_epoch(
			mean_error,
			sigma_error,
			mean_params,
			sigma_params,
			norm_sigmas
		)
		self._write_epoch_to_csv(
			mean_error,
			sigma_error,
			mean_params,
			sigma_params,
			norm_sigmas,
			mean_targets=mean_observed.values() if self.target_dict else None,
			sigma_targets=sigma_observed.values() if self.target_dict else None
		)
		


		self._mean_error_history.append(mean_error)
		self._sigma_error_history.append(sigma_error)

		for i, p in enumerate(self.parameters):
			self._mean_params_history[p].append(mean_params[i])
			self._sigma_params_history[p].append(sigma_params[i])
			self._norm_sigmas_history[p].append(norm_sigmas[i])

		if self.target_dict:
			for i, t in enumerate(self.target_dict):
				self._mean_target_history[t].append(mean_observed[t])
				self._sigma_target_history[t].append(sigma_observed[t])

		# if all solution directories are empty, remove epoch dir
		if len(os.listdir(os.path.dirname(solution_folder))) == 0:
			os.rmdir(os.path.dirname(solution_folder))
		return errors


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