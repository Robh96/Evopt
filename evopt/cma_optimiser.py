import cma
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from .base_optimiser import BaseOptimiser


class CmaesOptimiser(BaseOptimiser):
	"""
	Optimiser class that implements the CMA-ES algorithm.
	"""

	def setup_opt(self, epoch=None):
		"""
		Set up the CMA-ES optimiser.

		Args:
			epoch (int, optional): The epoch number to start from.
			Defaults to None.
		"""
		es = self.dir_manager.load_checkpoint(epoch)
		if es is None:
			opts = {
				'maxiter': self.n_epochs if self.n_epochs is not None else 1000000, # large number
				'seed': self.rand_seed,
				'popsize': self.batch_size,
				'bounds': [list(bound) for bound in zip(*self.norm_bounds)],
				'verbose': -9# 1 if self.verbose else -9
			}
			warnings.simplefilter("ignore", UserWarning)
			es = cma.CMAEvolutionStrategy(self.init_params, 1.0, opts)
			if self.verbose:
				print(f"Starting new CMAES run in directory {self.dir_manager.evolve_dir}")
		elif self.verbose:
			print(f"Continuing CMAES run from epoch {es.countiter} in directory {self.dir_manager.evolve_dir}")
		self.es = es
		self.current_epoch = self.es.countiter
		
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
		#sigmas = np.array([v[-1] for p, v in self.norm_sigmas.items() if v])
		sigmas = self.es.sigma * np.sqrt(np.diag(self.es.C))
		if len(sigmas) == 0:
			return False
		
		sigma_check = np.all(sigmas < self.sigma_threshold)
		epoch_check = self.n_epochs is not None and self.current_epoch >= self.n_epochs
		return sigma_check or epoch_check
	

	def optimise(self):
		"""
		Run the CMA-ES optimisation loop.
		Initialises the optimiser, then runs the optimisation loop
		until the termination criteria are met or the maximum number
		of epochs is reached.
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
		
		# Create results object
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
	"""Container for optimization results"""
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