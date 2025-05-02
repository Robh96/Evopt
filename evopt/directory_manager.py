"""Directory and file management for evolutionary optimization runs.

This module provides utilities for managing the directory structure, file organization,
and checkpointing functionality required for evolutionary optimization runs. It handles
creating consistent directory hierarchies, managing unique run identifiers, and providing
file paths for saving and retrieving optimization data.

The standard directory structure created is::

    evolve_<id>/                      # Root directory for a specific optimization run
    ├── epochs.csv                    # Aggregated statistics for each epoch
    ├── results.csv                   # Individual solution results
    ├── epochs/                       # Directory containing epoch-specific data
    │   ├── epoch0000/                # Data for epoch 0
    │   │   ├── solution0000/         # Data for solution 0 of epoch 0
    │   │   └── solution0001/         # Data for solution 1 of epoch 0
    │   └── epoch0001/                # Data for epoch 1
    ├── checkpoints/                  # Saved optimizer state for resumability
    │   ├── checkpoint_epoch0000.pkl
    │   └── checkpoint_epoch0001.pkl
    └── logs/                         # Log files for the optimization run

"""

import os
import pickle
import pandas as pd
from .utils import Logger

class DirectoryManager:
    """Manages directories for evolutionary optimization runs.
    
    This class handles the creation and management of directory structures
    for evolutionary optimization runs. It provides methods for directory 
    creation, solution organization, and checkpoint management.
    
    The directory structure is organized hierarchically to separate epochs
    and solutions, enabling clean organization of optimization results and
    facilitating analysis and visualization of the optimization process.
    
    Attributes:
        base_dir (str): Base directory where all optimization runs are stored.
        dir_id (int): Unique identifier for this optimization run.
        evolve_dir (str): Main directory for this specific optimization run.
        epochs_csv (str): Path to the CSV file containing epoch statistics.
        results_csv (str): Path to the CSV file containing individual solution results.
        epochs_dir (str): Directory containing epoch-specific data.
        checkpoint_dir (str): Directory for storing optimizer checkpoints.
        logs_dir (str): Directory for log files.
        logger (Logger): Logger instance for this optimization run.
        
    Example:
        >>> dm = DirectoryManager("./opt_results", dir_id=5)
        >>> dm.setup_directory()  # Creates the directory structure
        >>> epoch_folder = dm.create_epoch_folder(10)
        >>> solution_folder = dm.create_solution_folder(10, 3)
        >>> dm.save_checkpoint(optimizer_state, epoch=10)
        >>> optimizer_state = dm.load_checkpoint(epoch=10)
    """
    def __init__(self, base_dir: str, dir_id: int=None):
        """Initialize the DirectoryManager.
        
        Sets up paths for the directory structure and initializes the logger.
        If the specified directory already exists, it can be used to continue
        a previous optimization run.
        
        Args:
            base_dir: The base directory for all optimization runs.
            dir_id: The specific directory ID for this run. If None, a new ID 
                will be automatically generated. Default is None.
                
        Example:
            >>> # Create a new optimization run
            >>> dm = DirectoryManager("./optimization_results")
            >>> 
            >>> # Continue a specific run
            >>> dm = DirectoryManager("./optimization_results", dir_id=3)
        """
        self.base_dir = base_dir
        self.dir_id = self.get_dir_id(dir_id)
        self.evolve_dir = os.path.join(self.base_dir, f"evolve_{self.dir_id}")
        self.samples_dir = os.path.join(self.evolve_dir, f"samples")
        self.epochs_csv = os.path.join(self.evolve_dir, "epochs.csv")
        self.results_csv = os.path.join(self.evolve_dir, "results.csv")
        self.epochs_dir = os.path.join(self.evolve_dir, "epochs")
        self.checkpoint_dir = os.path.join(self.evolve_dir, "checkpoints")
        self.logs_dir = os.path.join(self.evolve_dir, "logs")
        self.logger = Logger(self.logs_dir)
        self.setup_directory()
        

    def get_dir_id(self, dir_id: int = None) -> int:
        """Determine an available directory ID.
    
        Finds an appropriate directory ID based on existing directories and
        the provided ID (if any). This ensures each optimization run has a
        unique identifier.
        
        Logic:
            
        - If dir_id is provided, use that ID.
        - If not provided, check existing evolve directories and:
        
            - If none exist, use ID 0.
            - Otherwise, find the smallest non-negative integer not already used.
        
        Args:
            dir_id: The directory ID to use if provided. Default is None.
            
        Returns:
            int: An available directory ID.
            
        Raises:
            FileNotFoundError: If base_dir does not exist.
            
        Example:
            >>> # Let the system choose an ID
            >>> dm = DirectoryManager("./results")
            >>> print(f"Assigned ID: {dm.dir_id}")
            
            >>> # Force a specific ID
            >>> dm = DirectoryManager("./results", dir_id=42)
            >>> print(f"Using ID: {dm.dir_id}")  # Will be 42
        """
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)

        files = [f for f in os.listdir(self.base_dir) if f.startswith("evolve_")]
        existing_ids = sorted([int(f.split("_")[-1]) for f in files if f.split("_")[-1].isdigit()])
        
        if dir_id is not None:
            return dir_id
        
        # Find the smallest missing ID
        if not existing_ids:
            return 0
        return next((i for i in range(max(existing_ids) + 2) if i not in existing_ids), 0)

    def setup_directory(self):
        """Create the main directory structure for the optimization run.
        
        Establishes the core directory hierarchy needed for an optimization run,
        including directories for epochs, checkpoints, and logs.
        
        Returns:
            None
            
        Example:
            >>> dm = DirectoryManager("./results")
            >>> dm.setup_directory()  # Creates all necessary directories
        """
        # Create the main directory structure
        os.makedirs(self.evolve_dir, exist_ok=True)
        os.makedirs(self.epochs_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

    def create_epoch_folder(self, epoch: int) -> str:
        """Create a folder for a specific optimization epoch.
        
        Creates a directory to store all data related to a specific epoch
        of the optimization process.
        
        Args:
            epoch: The epoch number.
            
        Returns:
            str: The path to the created epoch folder.
            
        Example:
            >>> dm = DirectoryManager("./results")
            >>> epoch_path = dm.create_epoch_folder(5)
            >>> print(f"Epoch directory: {epoch_path}")
            # Output: Epoch directory: ./results/evolve_0/epochs/epoch0005
        """
        epoch_folder = os.path.join(self.epochs_dir, f"epoch{epoch:0>4}")
        os.makedirs(epoch_folder, exist_ok=True)
        return epoch_folder

    def create_solution_folder(self, epoch: int, solution: int) -> str:
        """Create a folder for a specific solution within an epoch.
        
        Creates a directory to store data for a single solution evaluation 
        within a particular epoch. The solution folder is nested within
        the corresponding epoch folder.
        
        Args:
            epoch: The epoch number.
            solution: The solution number within the epoch.
            
        Returns:
            str: The path to the created solution folder.
            
        Example:
            >>> dm = DirectoryManager("./results")
            >>> solution_path = dm.create_solution_folder(2, 7)
            >>> print(f"Solution directory: {solution_path}")
            # Output: Solution directory: ./results/evolve_0/epochs/epoch0002/solution0007
        """
        # Create a folder for a specific solution within an epoch folder
        epoch_folder = self.create_epoch_folder(epoch)
        solution_folder = os.path.join(epoch_folder, f"solution{solution:0>4}")
        os.makedirs(solution_folder, exist_ok=True)
        return solution_folder
    

    def create_sample_folder(self, sample: int) -> str:
        """
        Create a folder to store data for a specific sample within an sample study.
        The created folder is nested within the corresponding evolve folder.
        
        Args:
            sample: The sample number within the exploratory study.
            
        Returns:
            str: The path to the created solution folder.
            
        Example:
            >>> dm = DirectoryManager("./results")
            >>> sample_path = dm.create_sample_folder(7)
            >>> print(f"sample directory: {sample_path}")
            # Output: sample directory: ./results/evolve_0/samples/sample0007
        """
        # Create a folder for a specific solution within an epoch folder
        sample_folder = os.path.join(self.samples_dir, f"sample{sample:0>4}")
        os.makedirs(sample_folder, exist_ok=True)
        return sample_folder
    
    def get_checkpoint_filepath(self, epoch: int) -> str:
        """Get the filepath for a checkpoint file for a specific epoch.
        
        Constructs the complete file path for saving or loading a checkpoint
        for the specified epoch.
        Checkpoint number corresponds to the latest complete epoch.
        
        Args:
            epoch: The latest complete epoch number.
            
        Returns:
            str: The full filepath for the checkpoint file.
            
        Example:
            >>> dm = DirectoryManager("./results")
            >>> checkpoint_path = dm.get_checkpoint_filepath(15)
            >>> print(f"Checkpoint file: {checkpoint_path}")
            # Output: Checkpoint file: ./results/evolve_0/checkpoints/checkpoint_epoch0015.pkl
        """
        return os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch:04d}.pkl")
    
    def save_checkpoint(self, data, epoch: int):
        """Save a checkpoint file for a specific epoch.
        
        Serializes and saves the optimizer state or other data to a checkpoint file
        for later resumption of the optimization process.
        
        Args:
            data: The data to save in the checkpoint file (typically optimizer state).
            epoch: The epoch number.
            
        Returns:
            None
            
        Raises:
            PermissionError: If the file cannot be written due to permission issues.
            pickle.PickleError: If the data cannot be serialized.
            
        Example:
            >>> dm = DirectoryManager("./results")
            >>> optimizer_state = {"params": [1.2, 3.4], "generation": 5}
            >>> dm.save_checkpoint(optimizer_state, epoch=5)
        """
        filepath = self.get_checkpoint_filepath(epoch)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_checkpoint(self, epoch: int = None):
        """Load a checkpoint file.
        
        Loads and deserializes optimizer state or other data from a checkpoint file.
        Can either load a specific epoch's checkpoint or the latest available checkpoint.
        
        Args:
            epoch: The specific epoch number to load the checkpoint from.
                If None, the latest checkpoint will be loaded. Default is None.
                
        Returns:
            Any: The data loaded from the checkpoint file, or None if no checkpoint is found.
            
        Raises:
            pickle.UnpicklingError: If the checkpoint file is corrupted or invalid.
            
        Example:
            >>> dm = DirectoryManager("./results")
            >>> # Load the latest checkpoint
            >>> latest_state = dm.load_checkpoint()
            >>> 
            >>> # Load a specific epoch's checkpoint
            >>> state = dm.load_checkpoint(epoch=10)
            >>> if state is not None:
            ...     print("Checkpoint loaded successfully")
            ... else:
            ...     print("No checkpoint found")
        """
        if epoch is not None:
            filepath = self.get_checkpoint_filepath(epoch)
        else:
            # If no epoch specified, try to find the latest checkpoint
            try:
                files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint")]
                if not files:
                    return None
                filepath = os.path.join(self.checkpoint_dir, max(files))
            except FileNotFoundError:
                # Directory might not exist yet
                return None
                
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
        except (pickle.UnpicklingError, EOFError):
            # Handle corrupted checkpoint files
            if self.logger:
                print(f"Warning: Checkpoint file {filepath} is corrupted and cannot be loaded.")
            return None
        
    def load_sample_history(self):
        """Load the sample history from the results CSV file."""
        completed_samples = set()
        if os.path.exists(self.results_csv):
            try:
                df = pd.read_csv(self.results_csv)
                if 'sample' in df.columns:
                    completed_samples = set(df['sample'].unique())
                    print(f"Loaded {len(completed_samples)} completed samples from {self.results_csv}.")
                else:
                    print(f"Warning: 'sample' column not found in {self.results_csv}.")
            
            except pd.errors.EmptyDataError:
                print(f"Results file {self.results_csv} is empty. Starting fresh.")
            
            except Exception as e:
                print(f"Error loading results file {self.results_csv}: {e}. Starting fresh.")
        return completed_samples
        
        