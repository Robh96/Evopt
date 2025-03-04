import os
import pickle
from contextlib import contextmanager
from .utils import Logger

class DirectoryManager:
	"""
	Manages directories for evolutionary optimization runs, including creating,
	navigating, and saving checkpoints.
	"""
	def __init__(self, base_dir: str, dir_id: int=None):
		"""
		Initialise the DirectoryManager.

		Args:
			base_dir (str): The base directory for all runs.
			dir_id (int, optional): The specific directory ID for this run.
				If None, a new ID will be generated. Defaults to None.
		"""
		self.base_dir = base_dir
		self.dir_id = self.get_dir_id(dir_id)
		self.evolve_dir = os.path.join(self.base_dir, f"evolve_{self.dir_id}")
		self.epochs_csv = os.path.join(self.evolve_dir, "epochs.csv")
		self.results_csv = os.path.join(self.evolve_dir, "results.csv")
		self.epochs_dir = os.path.join(self.evolve_dir, "epochs")
		self.checkpoint_dir = os.path.join(self.evolve_dir, "checkpoints")
		self.logs_dir = os.path.join(self.evolve_dir, "logs")
		self.logger = Logger(self.logs_dir)
		self.setup_directory()
		

	def get_dir_id(self, dir_id: int = None) -> int:
		"""
		Determine an available directory ID.

		If dir_id is provided:
			- Use it if no existing evolve directory with that ID exists.
			- If an evolve directory with that ID already exists, continue from that run.
		If no dir_id is provided:
			- Check the base_dir for existing evolve directories.
			- If none exist, return 0.
			- If multiple exist, find the smallest non-negative integer not in the list of existing IDs.

		Args:
			dir_id (int, optional): The directory ID to use. Defaults to None.

		Returns:
			int: An available directory ID.
		"""
		files = [f for f in os.listdir(self.base_dir) if f.startswith("evolve_")]
		existing_ids = sorted([int(f.split("_")[-1]) for f in files if f.split("_")[-1].isdigit()])
		
		if dir_id is not None:
			return dir_id
		
		# Find the smallest missing ID
		if not existing_ids:
			return 0
		return next((i for i in range(max(existing_ids) + 2) if i not in existing_ids), 0)



	@contextmanager
	def working_directory(self, path: str):
		"""
		A context manager which changes the working directory to the given
		path, and then changes it back to its previous value on exit.

		Args:
			path (str): The path to change the working directory to.
		"""
		prev_cwd = os.getcwd()
		os.makedirs(path, exist_ok=True) # Ensure the directory exists
		os.chdir(path)
		try:
			yield
		finally:
			os.chdir(prev_cwd)

	def setup_directory(self):
		"""
		Create the main directory structure for the evolutionary optimization run.
		"""
		# Create the main directory structure
		os.makedirs(self.evolve_dir, exist_ok=True)
		os.makedirs(self.epochs_dir, exist_ok=True)
		os.makedirs(self.checkpoint_dir, exist_ok=True)

	def create_epoch_folder(self, epoch: int) -> str:
		"""
		Create a folder for a specific epoch.

		Args:
			epoch (int): The epoch number.

		Returns:
			str: The path to the created epoch folder.
		"""
		# Create a folder for a specific epoch
		epoch_folder = os.path.join(self.epochs_dir, f"epoch{epoch:0>4}")
		os.makedirs(epoch_folder, exist_ok=True)
		return epoch_folder

	def create_solution_folder(self, epoch: int, solution: int) -> str:
		"""
		Create a folder for a specific solution within an epoch folder.

		Args:
			epoch (int): The epoch number.
			solution (int): The solution number.

		Returns:
			str: The path to the created solution folder.
		"""
		# Create a folder for a specific solution within an epoch folder
		epoch_folder = self.create_epoch_folder(epoch)
		solution_folder = os.path.join(epoch_folder, f"solution{solution:0>4}")
		os.makedirs(solution_folder, exist_ok=True)
		return solution_folder
	
	def get_checkpoint_filepath(self, epoch: int) -> str:
		"""
		Get the filepath for a checkpoint file for a specific epoch.

		Args:
			epoch (int): The epoch number.

		Returns:
			str: The filepath for the checkpoint file.
		"""
		return os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch:04d}.pkl")
	
	def save_checkpoint(self, data, epoch: int):
		"""
		Save a checkpoint file for a specific epoch.

		Args:
			data: The data to save in the checkpoint file.
			epoch (int): The epoch number.
		"""
		filepath = self.get_checkpoint_filepath(epoch)
		with open(filepath, 'wb') as f:
			pickle.dump(data, f)

	def load_checkpoint(self, epoch: int = None):
		"""
		Load a checkpoint file.

		Args:
			epoch (int, optional): The epoch number to load the checkpoint from.
				If None, the latest checkpoint will be loaded. Defaults to None.

		Returns:
			The data loaded from the checkpoint file, or None if no checkpoint is found.
		"""
		if epoch is not None:
			filepath = self.get_checkpoint_filepath(epoch)
		else:
			files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint")]
			if not files:
				return None
			filepath = os.path.join(self.checkpoint_dir, max(files))
		try:
			with open(filepath, 'rb') as f:
				return pickle.load(f)
		except FileNotFoundError:
			return None