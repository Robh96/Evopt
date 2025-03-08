# contains logger, and helpers (csv_writer, format_array, convert_to_native)
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
    """Enum representing different execution environments"""
    LOCAL = auto()
    SLURM = auto()
    PBS = auto()
    LSF = auto()
    
def detect_environment():
    """Detect the execution environment"""
    if 'SLURM_JOB_ID' in os.environ:
        return ExecutionEnvironment.SLURM
    elif 'PBS_JOBID' in os.environ:
        return ExecutionEnvironment.PBS
    elif 'LSB_JOBID' in os.environ:
        return ExecutionEnvironment.LSF
    return ExecutionEnvironment.LOCAL

def get_available_cpus():
    """Get number of CPUs available, accounting for HPC environment variables"""
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
    """A context manager for changing working directory temporarily"""
    prev_cwd = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

class SlurmJobManager:
    """Manages SLURM job submissions"""
    
    @staticmethod
    def submit_job(script_content, job_name, cpus_per_task, output_dir, 
                  memory_gb=None, wall_time="01:00:00", qos=None):
        """Submit a job to SLURM scheduler"""
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
    def wait_for_job(job_id, check_interval=10):
        """Wait for a SLURM job to complete"""
        while True:
            cmd = ["squeue", "-j", str(job_id), "-h"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            # If no output, job is done
            if not result.stdout.strip():
                return
            time.sleep(check_interval)
    
    @staticmethod
    def cancel_job(job_id):
        """Cancel a SLURM job"""
        subprocess.run(["scancel", str(job_id)], check=False)

class ProcessPoolManager:
    """Manages process pools across different execution environments"""
    
    def __init__(
            self,
            max_workers=1,
            cores_per_worker=1,
            memory_gb_per_worker=4,
            wall_time: str="01:00:00", # Default walltime (hh:mm:ss) for SLURM jobs
            qos=None,  # Quality of Service for SLURM jobs
            ):
        
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
        """Initialize the appropriate executor based on environment"""
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
                mp_context=mp.get_context("spawn")
            )
            return self._executor
        return None  # No executor means fall back to serial processing
    
    def cleanup(self):
        """Clean up resources"""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        
        # Cancel any pending SLURM jobs
        for job_id in self._job_ids:
            if self.env == ExecutionEnvironment.SLURM:
                SlurmJobManager.cancel_job(job_id)
        self._job_ids = []
    
    def __del__(self):
        """Ensure resources are cleaned up"""
        self.cleanup()

def convert_to_native(value):
    """
    Convert a value to a native Python type for serialization.

    Args:
        value: The value to convert.

    Returns:
        The converted value.
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
    """
    Format a numpy array into a string with a specified precision.

    Args:
        arr (np.ndarray): The array to format.
        precision (int, optional): The number of decimal places to include. Defaults to 3.

    Returns:
        str: A string representation of the array.
    """
    return ", ".join(f"{x:.{precision}f}" for x in arr)

def write_to_csv(data, csv_path, sort_columns=None):
    """
    Write a dictionary of data to a CSV file with optional sorting.

    Args:
        data (dict): The data to write.
        csv_path (str): The path to the CSV file.
        sort_columns (list, optional): Columns to sort by. Defaults to None.
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

def extend_dict(master_dict, slave_dict):
    """
    Extends the observed_dict with values from loss_observed_dict.
    If a key exists in both dictionaries, the values from loss_observed_dict
    are appended to the list of values in observed_dict. If a key only
    exists in loss_observed_dict, it is added to observed_dict.

    Args:
        observed_dict (dict): The dictionary to extend.
        loss_observed_dict (dict): The dictionary containing new values.
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
    """
    A simple logger that writes messages to both the terminal and a log file.
    """
    def __init__(self, log_dir, log_file="logfile.log"):
        """
        Initialise the Logger.

        Args:
            log_dir (str): The directory to store the log file.
            log_file (str, optional): The name of the log file. Defaults to "logfile.log".
        """
        self.terminal = sys.stdout
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.log = None

    def __enter__(self):
        """
        Enter the context and redirect stdout to both the terminal and the log file.
        """
        self.log = open(self.log_path, "a")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context and restore stdout to the terminal.
        """
        sys.stdout = self.terminal
        if self.log:
            self.log.close()
    
    def __getattr__(self, attr):
        """
        Delegate attribute access to the terminal.

        Args:
            attr (str): The attribute to access.

        Returns:
            The attribute from the terminal.
        """
        return getattr(self.terminal, attr)
    
    def write(self, message):
        """
        Write a message to both the terminal and the log file.

        Args:
            message (str): The message to write.
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

    def flush(self):
        """
        Flush the buffers of both the terminal and the log file.
        """
        self.terminal.flush()
        self.log.flush()