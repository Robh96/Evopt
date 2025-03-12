"""Utility functions and classes for evolutionary optimization.

This module provides various utility functions and classes for supporting evolutionary optimization
processes in different environments, including HPC clusters and local machines. It includes tools for
environment detection, process management, file operations, dictionary manipulation, and logging.

The module supports different execution environments:
- Local execution
- SLURM cluster execution
- PBS/Torque cluster execution
- LSF cluster execution

Example:
    Basic usage of the process pool manager:
    
    >>> from evopt.utils import ProcessPoolManager
    >>> manager = ProcessPoolManager(max_workers=4)
    >>> executor = manager.initialize()
    >>> # Use executor for parallel processing
    >>> manager.cleanup()
    
    Using the logging utility:
    
    >>> with Logger("./logs") as logger:
    >>>     print("This will be logged to both console and file")
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import multiprocessing as mp
import concurrent.futures
import subprocess
import tempfile
import time
from contextlib import contextmanager
from enum import Enum, auto

# Set multiprocessing start method once during import
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set
    pass

class ExecutionEnvironment(Enum):
    """Enum representing different execution environments.
    
    This enumeration defines the different types of execution environments
    that the code can run in, from local machines to various HPC systems.
    
    Attributes:
        LOCAL: Local machine execution.
        SLURM: SLURM-based cluster execution.
        PBS: PBS/Torque cluster execution.
        LSF: LSF cluster execution.
    """

    LOCAL = auto()
    SLURM = auto()
    PBS = auto()
    LSF = auto()
    
def detect_environment() -> ExecutionEnvironment:
    """Detect the current execution environment based on environment variables.
    
    This function examines environment variables to determine whether the code
    is running on a local machine or in a cluster environment (SLURM, PBS, LSF).
    
    Returns:
        ExecutionEnvironment: The detected execution environment.
        
    Example:
        >>> env = detect_environment()
        >>> if env == ExecutionEnvironment.SLURM:
        >>>     print("Running in a SLURM environment")
    """

    if 'SLURM_JOB_ID' in os.environ:
        return ExecutionEnvironment.SLURM
    elif 'PBS_JOBID' in os.environ:
        return ExecutionEnvironment.PBS
    elif 'LSB_JOBID' in os.environ:
        return ExecutionEnvironment.LSF
    return ExecutionEnvironment.LOCAL

def get_available_cpus() -> int:
    """Get number of CPUs available, accounting for HPC environment variables.
    
    This function determines the number of available CPUs based on the current
    execution environment, checking relevant environment variables for 
    different HPC systems before falling back to the system CPU count.
    
    Returns:
        int: The number of available CPUs.
        
    Example:
        >>> num_cpus = get_available_cpus()
        >>> print(f"Using {num_cpus} CPUs for computation")
    """

    env = detect_environment()
    
    # SLURM-specific environment variables
    if env == ExecutionEnvironment.SLURM:
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            return int(os.environ['SLURM_CPUS_PER_TASK'])
        elif 'SLURM_NTASKS' in os.environ:
            return int(os.environ['SLURM_NTASKS'])
        elif 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
            # This might be complex like "16(x2),12" - take the first number
            cpus_str = os.environ['SLURM_JOB_CPUS_PER_NODE'].split('(')[0]
            return int(cpus_str)
    
    # PBS-specific environment variables
    elif env == ExecutionEnvironment.PBS:
        if 'PBS_NP' in os.environ:
            return int(os.environ['PBS_NP'])
    
    # Generic OpenMP environment variable
    if 'OMP_NUM_THREADS' in os.environ:
        return int(os.environ['OMP_NUM_THREADS'])
    
    # Fall back to CPU count if no environment variables are set
    return mp.cpu_count()

@contextmanager
def working_directory(path):
    """A context manager for temporarily changing the working directory.
    
    This context manager allows for temporarily changing the working directory
    and automatically restoring it when exiting the context.
    
    Args:
        path: Path to the directory to change to.
        
    Yields:
        None
        
    Example:
        >>> with working_directory('/path/to/dir'):
        >>>     # Do something in the new directory
        >>>     pass
    """

    prev_cwd = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

class SlurmJobManager:
    """Manages job submissions to SLURM workload manager.
    
    This class provides static methods for submitting, monitoring, and canceling jobs
    on SLURM-based HPC clusters. It handles script creation, job submission,
    and job status monitoring.
    
    Examples:
        Submit a simple job:
        
        >>> job_id = SlurmJobManager.submit_job(
        ...     "python my_script.py", 
        ...     "optimization", 
        ...     4, 
        ...     "./output"
        ... )
        >>> print(f"Job submitted with ID: {job_id}")
        
        Wait for job completion:
        
        >>> SlurmJobManager.wait_for_job(job_id)
        >>> print("Job completed")
    """
    
    @staticmethod
    def submit_job(
        script_content: str,
        job_name: str,
        cpus_per_task: int,
        output_dir: str,
        memory_gb: float = None,
        wall_time: str = "01:00:00",
        qos: str = None
    ) -> int:
        """Submit a job to the SLURM scheduler.
        
        Creates a temporary SLURM batch script with the specified parameters and
        submits it to the SLURM workload manager.
        
        Args:
            script_content: The script content to be executed.
            job_name: Name for the SLURM job.
            cpus_per_task: Number of CPUs to allocate per task.
            output_dir: Directory for SLURM output and error files.
            memory_gb: Memory allocation in gigabytes. Default is None (uses SLURM default).
            wall_time: Maximum wall time in format "HH:MM:SS". Default is "01:00:00".
            qos: Quality of Service to request. Default is None.
            
        Returns:
            int: The SLURM job ID.
            
        Raises:
            subprocess.SubprocessError: If job submission fails.
            ValueError: If job ID cannot be parsed from SLURM response.
            
        Example:
            >>> job_id = SlurmJobManager.submit_job(
            ...     "python run_optimization.py",
            ...     "evopt_job", 
            ...     4, 
            ...     "./slurm_logs",
            ...     memory_gb=16
            ... )
        """

        # Create a temporary script file
        fd, path = tempfile.mkstemp(suffix='.sh')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={job_name}\n")
                f.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
                if memory_gb:
                    f.write(f"#SBATCH --mem={memory_gb}G\n")
                f.write(f"#SBATCH --time={wall_time}\n")
                if qos:
                    f.write(f"#SBATCH --qos={qos}\n")
                f.write(f"#SBATCH --output={output_dir}/slurm_%j.out\n")
                f.write(f"#SBATCH --error={output_dir}/slurm_%j.err\n")
                f.write(f"\n")
                f.write(f"export OMP_NUM_THREADS={cpus_per_task}\n")
                f.write(f"{script_content}\n")
            
            # Submit the job
            cmd = ["sbatch", path]
            result = subprocess.check_output(cmd, text=True)
            # Parse job ID from output
            job_id = int(result.strip().split()[-1])
            return job_id
        finally:
            os.unlink(path)  # Clean up temp file
    
    @staticmethod
    def wait_for_job(job_id: int, check_interval: int = 10) -> None:
        """Wait for a SLURM job to complete.
        
        Polls the SLURM queue periodically until the specified job
        is no longer found in the queue, indicating completion.
        
        Args:
            job_id: The SLURM job ID to monitor.
            check_interval: Seconds to wait between checks. Default is 10.
            
        Returns:
            None
            
        Raises:
            subprocess.SubprocessError: If checking job status fails.
            
        Example:
            >>> job_id = SlurmJobManager.submit_job(...)
            >>> SlurmJobManager.wait_for_job(job_id, check_interval=30)
            >>> print("Job finished")
        """

        while True:
            cmd = ["squeue", "-j", str(job_id), "-h"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            # If no output, job is done
            if not result.stdout.strip():
                return
            time.sleep(check_interval)
    
    @staticmethod
    def cancel_job(job_id: int) -> None:
        """Cancel a running SLURM job.
        
        Sends a cancellation request for the specified job ID.
        
        Args:
            job_id: The SLURM job ID to cancel.
            
        Returns:
            None
            
        Example:
            >>> job_id = SlurmJobManager.submit_job(...)
            >>> # Cancel the job if needed
            >>> SlurmJobManager.cancel_job(job_id)
        """

        subprocess.run(["scancel", str(job_id)], check=False)

class ProcessPoolManager:
    """Manages process pools across different execution environments.
    
    This class provides a unified interface for parallel processing across
    different computing environments (local machine or HPC clusters).
    It automatically detects the environment and sets up the appropriate
    executor for parallel task processing.
    
    Attributes:
        env: The detected execution environment.
        cores_per_worker: Number of CPU cores allocated to each worker.
        memory_gb_per_worker: Memory (GB) allocated to each worker.
        wall_time: Maximum job runtime for HPC jobs.
        qos: Quality of Service for SLURM jobs.
        max_workers: Maximum number of concurrent workers.
        
    Examples:
        Basic usage on a local machine:
        
        >>> manager = ProcessPoolManager(max_workers=4)
        >>> with manager.initialize() as executor:
        ...     results = list(executor.map(my_function, my_inputs))
        
        Usage with more specific resource requirements:
        
        >>> manager = ProcessPoolManager(
        ...     max_workers=8,
        ...     cores_per_worker=4,
        ...     memory_gb_per_worker=16,
        ...     wall_time="08:00:00"
        ... )
        >>> executor = manager.initialize()
        >>> try:
        ...     futures = [executor.submit(my_function, arg) for arg in my_inputs]
        ...     for future in concurrent.futures.as_completed(futures):
        ...         result = future.result()
        ... finally:
        ...     manager.cleanup()
    """
    
    def __init__(
        self,
        max_workers: int = 1,
        cores_per_worker: int = 1,
        memory_gb_per_worker: float = 4,
        wall_time: str = "01:00:00",
        qos: str = None
    ):
        """Initialize the ProcessPoolManager.
        
        Args:
            max_workers: Maximum number of concurrent worker processes.
                Default is 1.
            cores_per_worker: CPU cores allocated to each worker process.
                Default is 1.
            memory_gb_per_worker: Memory in GB allocated to each worker.
                Default is 4 GB.
            wall_time: Maximum job runtime in format "HH:MM:SS".
                Default is "01:00:00" (1 hour).
            qos: Quality of Service for SLURM jobs. Default is None.
                
        Example:
            >>> manager = ProcessPoolManager(
            ...     max_workers=4,
            ...     cores_per_worker=2,
            ...     memory_gb_per_worker=8
            ... )
        """

        self.env = detect_environment()
        self.cores_per_worker = cores_per_worker
        self.memory_gb_per_worker = memory_gb_per_worker
        self.wall_time = wall_time
        self.qos = qos
        self.max_workers = max_workers 
        self._executor = None
        self._file_lock = mp.Lock()  # For synchronizing file access
        self._job_ids = []  # For tracking submitted HPC jobs
    
    def initialize(self):
        """Initialize the appropriate executor based on the execution environment.
        
        This method detects the current execution environment and creates the
        appropriate type of executor. For local environments, it creates a
        ProcessPoolExecutor. For SLURM environments, it creates a SlurmExecutor
        if available, otherwise falls back to local processing.
        
        Returns:
            concurrent.futures.Executor or None: The initialized executor object,
                or None if max_workers <= 1 (serial processing).
                
        Example:
            >>> manager = ProcessPoolManager(max_workers=4)
            >>> executor = manager.initialize()
            >>> if executor:
            ...     futures = [executor.submit(func, i) for i in range(10)]
        """
        
        if os.environ.get("EVOPT_WORKER", "0") == "1":
            if self.verbose:
                print("Detected nested worker process - batch processing should be avoided")
            raise RuntimeError(
                "Attempted to create a process pool from within a worker process. "
                "This suggests a design issue in the code - worker processes should "
                "only evaluate single solutions and not start their own batches."
            )

        if self._executor is not None:
            return self._executor
        
        if self.max_workers <= 1:
            return None  # No need for executor if only one worker
        
        # SLURM-specific environment variables
        if self.env == ExecutionEnvironment.SLURM:
            try:
                from .slurm_executor import SlurmExecutor
                self._executor = SlurmExecutor(
                    max_workers=self.max_workers,
                    cores_per_worker=self.cores_per_worker,
                    memory_gb_per_worker=self.memory_gb_per_worker,
                    wall_time=self.wall_time,
                    qos=self.qos
                )
                return self._executor
            except ImportError:
                print("Warning: SLURM environment detected but SLURM executor not available.")
                print("Falling back to local processing pool.")

        if self.env == ExecutionEnvironment.LOCAL:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context("spawn"),
                initializer=worker_init
            )
            return self._executor
        return None  # No executor means fall back to serial processing
    
    def check_workers_health(self):
        """Check if workers are healthy and replace any dead ones"""
        if hasattr(self._executor, '_processes'):
            for pid, process in list(self._executor._processes.items()):
                if not process.is_alive():
                    # Worker died - allow executor to replace it
                    print(f"Worker process {pid} died unexpectedly")

    def cleanup(self):
        """Clean up resources used by the executor.
        
        Shuts down the executor if it exists and cancels any pending SLURM jobs.
        This method should be called when the executor is no longer needed to
        ensure proper resource cleanup.
        
        Returns:
            None
            
        Example:
            >>> manager = ProcessPoolManager(max_workers=4)
            >>> executor = manager.initialize()
            >>> try:
            ...     # Use executor for parallel processing
            ...     pass
            ... finally:
            ...     manager.cleanup()
        """
        
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        # Cancel any pending SLURM jobs
        for job_id in self._job_ids:
            if self.env == ExecutionEnvironment.SLURM:
                SlurmJobManager.cancel_job(job_id)
        self._job_ids = []
    
    def __del__(self):
        """Ensure resources are cleaned up when the instance is garbage collected."""
        self.cleanup()


def worker_init():
    """Initialize worker processes as EVOPT_WORKER."""

    os.environ["EVOPT_WORKER"] = "1"

def convert_to_native(value):
    """Convert a value to a native Python type for serialization.

    Converts NumPy types, nested lists, and dictionaries to their native Python
    equivalents. This is particularly useful for preparing data for JSON serialization
    or other formats that don't support NumPy types.

    Args:
        value (Any): The value to convert. Can be a NumPy type, list, dict, or None.

    Returns:
        Any: The converted value in native Python format.
        
    Example:
        >>> import numpy as np
        >>> data = {'value': np.float64(3.14159), 'array': np.array([1.0, 2.0, 3.0])}
        >>> convert_to_native(data)
        {'value': 3.142, 'array': [1.0, 2.0, 3.0]}
    """

    if isinstance(value, (np.float64, float)):
        return round(float(value), 3)
    elif isinstance(value, list):
        return [convert_to_native(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_to_native(v) for k, v in value.items()}
    elif value is None:
        return 'None'
    return value

def format_array(arr, precision=3):
    """Format a numpy array into a string with a specified precision.

    Creates a comma-separated string representation of the array values,
    with each value formatted to the specified precision.

    Args:
        arr (np.ndarray or list): The array to format.
        precision (int, optional): The number of decimal places to include. Default is 3.

    Returns:
        str: A string representation of the array.
        
    Example:
        >>> import numpy as np
        >>> arr = np.array([1.2345, 5.6789, 9.8765])
        >>> format_array(arr, precision=2)
        '1.23, 5.68, 9.88'
    """

    return ", ".join(f"{x:.{precision}f}" for x in arr)

def write_to_csv(data, csv_path, sort_columns=None):
    """Write a dictionary of data to a CSV file with optional sorting.

    Appends a row of data to an existing CSV file or creates a new one.
    The function handles numeric conversion and supports sorting by specified columns.

    Args:
        data (dict): The data to write as a row in the CSV.
        csv_path (str): The path to the CSV file.
        sort_columns (list, optional): Column names to sort the data by. Default is None.
        
    Returns:
        None
        
    Raises:
        OSError: If the file cannot be created or written to.
        
    Example:
        >>> data = {'epoch': 1, 'error': 0.023, 'param1': 0.5, 'param2': 1.2}
        >>> write_to_csv(data, 'results.csv', sort_columns=['epoch'])
    """

    data = {k: convert_to_native(v) for k, v in data.items()}
    df_row = pd.DataFrame([data])
    
    if os.path.isfile(csv_path):
        # Read existing CSV, append new data, sort, and rewrite
        try:
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_row], ignore_index=True)
            
            if sort_columns:
                df_combined = df_combined.sort_values(by=sort_columns)
                
            df_combined.to_csv(csv_path, mode='w', header=True, index=False)
        except pd.errors.EmptyDataError:
            # File exists but is empty
            df_row.to_csv(csv_path, mode='w', header=True, index=False)
    else:
        # Creating new file
        df_row.to_csv(csv_path, mode='w', header=True, index=False)

def extend_dict(master_dict: dict, slave_dict: dict) -> None:
    """Merge dictionary keys and extend values as lists.
    
    If a key exists in both dictionaries, the values from slave_dict
    are appended to the list of values in master_dict. If a key only
    exists in slave_dict, it is added to master_dict.
    
    This function modifies master_dict in-place.

    Args:
        master_dict (dict): The dictionary to extend (modified in-place).
        slave_dict (dict): The dictionary containing values to add.
        
    Returns:
        None: The master_dict is modified in-place.
        
    Example:
        >>> master = {'a': [1, 2], 'b': 3}
        >>> slave = {'a': 4, 'b': [5, 6], 'c': 7}
        >>> extend_dict(master, slave)
        >>> print(master)
        {'a': [1, 2, 4], 'b': [3, 5, 6], 'c': 7}
    """

    for key, value in slave_dict.items():
        if key in master_dict:
            if isinstance(master_dict[key], list):
                master_dict[key].extend(value if isinstance(value, list) else [value])
            else:
                master_dict[key] = [master_dict[key]] + (value if isinstance(value, list) else [value])
        else:
            master_dict[key] = value if isinstance(value, list) else [value]


class Logger:
    """A simple logger that writes messages to both the terminal and a log file.
    
    This class redirects standard output to both the console and a log file,
    adding timestamps to each line. It's designed to be used as a context manager.
    
    Attributes:
        terminal: The original stdout stream.
        log_path: Path to the log file.
        log: File handle for the log file.
        
    Examples:
        Basic usage:
        
        >>> with Logger("./logs", "my_run.log") as logger:
        ...     print("This message goes to both console and log file")
        
        Simple message logging:
        
        >>> logger = Logger("./logs")
        >>> with logger:
        ...     for i in range(5):
        ...         print(f"Processing item {i}")
    """

    def __init__(self, log_dir: str, log_file: str = "logfile.log"):
        """Initialize the logger.

        Args:
            log_dir (str): Directory to store the log file.
            log_file (str, optional): Name of the log file. Default is "logfile.log".
            
        Example:
            >>> logger = Logger("./experiment_logs", "run_1.log")
        """

        self.terminal = sys.stdout
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.log = None

    def __enter__(self):
        """Enter the context and redirect stdout to both terminal and log file.
        
        This method is called when entering a 'with' statement. It redirects
        stdout to this Logger instance, which writes to both the console and
        the log file.
        
        Returns:
            Logger: The logger instance.
            
        Example:
            >>> with Logger("./logs") as logger:
            ...     print("This is logged")
        """

        self.log = open(self.log_path, "a")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore stdout to the terminal.
        
        This method is called when exiting a 'with' statement. It restores
        the original stdout and closes the log file.
        
        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred. 
            exc_tb: Exception traceback if an exception occurred.
            
        Returns:
            None
        """

        sys.stdout = self.terminal
        if self.log:
            self.log.close()
    
    def __getattr__(self, attr):
        """Delegate attribute access to the terminal.
        
        This allows the Logger to properly proxy attribute access to the
        underlying terminal object, ensuring compatibility with code that
        expects stdout to have certain attributes.

        Args:
            attr (str): The attribute name to access.

        Returns:
            Any: The attribute from the terminal object.
            
        Raises:
            AttributeError: If the terminal doesn't have the requested attribute.
        """

        return getattr(self.terminal, attr)
    
    def write(self, message: str) -> None:
        """Write a message to both the terminal and the log file.
        
        Adds a timestamp to each non-empty line before writing to the log file.
        The original message (without timestamps) is written to the terminal.

        Args:
            message (str): The message to write.
            
        Returns:
            None
        """

        # Prepend the current date and time to each line in the message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = message.splitlines(True)  # Keep the newline characters
        for line in lines:
            if line.strip():  # Only add timestamp to non-empty lines
                formatted_message = f"{timestamp} - {line}"
            else:
                formatted_message = line
            if self.log:
                self.log.write(formatted_message)    
            self.terminal.write(line)
        self.flush()

    def flush(self) -> None:
        """Flush the buffers of both the terminal and the log file.
        
        This ensures that any buffered output is written immediately.
        
        Returns:
            None
        """
        
        self.terminal.flush()
        self.log.flush()