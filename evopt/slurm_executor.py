"""SLURM-based parallel execution for evolutionary optimization tasks.

This module provides a concurrent.futures.Executor implementation that submits tasks
to a SLURM cluster for parallel execution. It handles job submission, monitoring,
and result retrieval, providing a seamless interface for distributed computing on
HPC clusters running SLURM workload manager.

The SlurmExecutor integrates with the Python concurrent.futures API, allowing code
written for ProcessPoolExecutor to run on SLURM clusters with minimal changes.

Example:
    Basic usage submitting a function to SLURM:
    
    >>> from evopt.slurm_executor import SlurmExecutor
    >>> executor = SlurmExecutor(max_workers=4, cores_per_worker=2, memory_gb=8)
    >>> future = executor.submit(my_function, arg1, arg2)
    >>> result = future.result()  # Blocks until the job completes
    >>> executor.shutdown()
    
    Using with concurrent.futures API:
    
    >>> with SlurmExecutor(max_workers=10) as executor:
    ...     results = list(executor.map(my_function, my_args))
"""

import os
import time
import pickle
import concurrent.futures
import tempfile
import threading
from .utils import SlurmJobManager

class SlurmExecutor(concurrent.futures.Executor):
    """Executor implementation that submits tasks as SLURM jobs.
    
    This class provides a concurrent.futures.Executor interface for executing tasks
    on a SLURM cluster. It handles job submission, monitoring, and result retrieval,
    allowing for efficient parallel execution across multiple compute nodes.
    
    The executor maintains a pool of worker slots limited by max_workers, submitting
    new jobs as slots become available and collecting results asynchronously.
    
    Attributes:
        max_workers (int): Maximum number of concurrent SLURM jobs.
        cores_per_worker (int): Number of CPU cores to request per job.
        memory_gb (float): Memory in GB to request per job.
        wall_time (str): Maximum wall time for each job in format "HH:MM:SS".
        qos (str or None): Quality of Service to request from SLURM.
        job_manager (SlurmJobManager): Manager for SLURM job operations.
        base_dir (str): Temporary directory for job files and results.
        
    Examples:
        Submit multiple tasks and process results as they complete:
        
        >>> executor = SlurmExecutor(max_workers=4)
        >>> futures = [executor.submit(process_data, data_chunk) for data_chunk in data]
        >>> for future in concurrent.futures.as_completed(futures):
        ...     try:
        ...         result = future.result()
        ...         print(f"Task completed with result: {result}")
        ...     except Exception as e:
        ...         print(f"Task failed: {e}")
        >>> executor.shutdown()
    """
    
    def __init__(
            self,
            max_workers: int = 1,
            cores_per_worker: int = 1,
            memory_gb: float = 4,
            wall_time: str = "1:00:00",
            qos: str = None
        ):
        """Initialize the SlurmExecutor with specified resources.
        
        Args:
            max_workers: Maximum number of concurrent SLURM jobs. Default is 1.
            cores_per_worker: Number of CPU cores to request per job. Default is 1.
            memory_gb: Memory in GB to request per job. Default is 4 GB.
            wall_time: Maximum wall time for jobs in format "HH:MM:SS". Default is "1:00:00".
            qos: Quality of Service to request from SLURM. Default is None.
            
        Example:
            >>> # Create an executor with 8 concurrent jobs, each with 4 cores and 16GB
            >>> executor = SlurmExecutor(
            ...     max_workers=8, 
            ...     cores_per_worker=4, 
            ...     memory_gb=16, 
            ...     wall_time="4:00:00"
            ... )
        """

        self.max_workers = max_workers
        self.cores_per_worker = cores_per_worker
        self.memory_gb = memory_gb
        self.wall_time = wall_time
        self.qos = qos
        
        self.job_manager = SlurmJobManager()
        self._shutdown = False
        self._jobs = {}  # job_id -> (future, result_path)
        
        # Create base temp directory for job results
        self.base_dir = tempfile.mkdtemp(prefix="evopt_slurm_")
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Semaphore to limit concurrent jobs
        self._job_semaphore = threading.Semaphore(self.max_workers)
        
    def submit(self, fn, *args, **kwargs):
        """Submit a function for execution as a SLURM job.
        
        This method submits a function and its arguments to be executed as a SLURM job.
        It creates a temporary directory for the job, pickles the function and arguments,
        generates a Python script to run in the SLURM environment, and submits the job.
        
        The method returns a Future object that can be used to check the status and
        retrieve the result of the job.
        
        Args:
            fn (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            concurrent.futures.Future: A Future representing the execution of the function.
            
        Raises:
            RuntimeError: If the executor has been shut down.
            Exception: Any exception raised during job submission is set on the future.
            
        Example:
            >>> def my_task(x, y, z=1):
            ...     return x * y * z
            >>> 
            >>> executor = SlurmExecutor(max_workers=4)
            >>> future = executor.submit(my_task, 2, 3, z=4)
            >>> result = future.result()  # Wait for the job to complete
            >>> print(result)  # Should print 24
        """

        if self._shutdown:
            raise RuntimeError("Executor is shutdown")
            
        # Wait for a job slot to become available
        self._job_semaphore.acquire()
        
        # Create future
        future = concurrent.futures.Future()
        
        # Extract working directory from args if it's the BaseOptimizer._evaluate_solution_worker
        # The solution_folder is the 4th argument (index 3) in the args tuple
        original_working_dir = None
        if hasattr(fn, '__name__') and fn.__name__ == '_evaluate_solution_worker' and len(args) > 0:
            # args[0] is the args tuple passed to _evaluate_solution_worker
            if isinstance(args[0], tuple) and len(args[0]) >= 4:
                original_working_dir = args[0][3]  # Extract solution_folder

        # Create unique job directory
        job_id = str(int(time.time() * 1000)) + str(hash(str(args) + str(kwargs)) % 10000)
        job_dir = os.path.join(self.base_dir, f"job_{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        
        # Pickle function and arguments
        input_file = os.path.join(job_dir, "task.pkl")
        output_file = os.path.join(job_dir, "result.pkl")
        
        with open(input_file, 'wb') as f:
            pickle.dump((fn, args, kwargs, original_working_dir), f)
        
        # Create the Python script that will run in SLURM
        script = f"""
import os
import sys
import pickle
import traceback

# Ensure the package is in the path (adjust as needed)
sys.path.insert(0, "{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")

# Load the pickled function and arguments
with open("{input_file}", 'rb') as f:
    fn, args, kwargs, original_working_dir = pickle.load(f)

# Execute the function in the correct directory if specified
try:
    if original_working_dir and os.path.exists(original_working_dir):
        # Change to the original working directory
        original_cwd = os.getcwd()
        os.chdir(original_working_dir)
        try:
            result = fn(*args, **kwargs)
            success = True
            error = None
        finally:
            # Restore original directory
            os.chdir(original_cwd)
    else:
        # No working directory specified or doesn't exist, run as-is
        result = fn(*args, **kwargs)
        success = True
        error = None
except Exception as e:
    result = None
    success = False
    error = traceback.format_exc()

# Save the result
with open("{output_file}", 'wb') as f:
    pickle.dump((success, result, error), f)

# Create a completion marker file
with open("{os.path.join(job_dir, 'COMPLETED')}", 'w') as f:
    f.write('done')
"""

        runner_path = os.path.join(job_dir, "runner.py")
        with open(runner_path, 'w') as f:
            f.write(script)
        
        # Create the job submission script
        job_script = f"python {runner_path}"
        
        # Submit the job
        try:
            slurm_job_id = self.job_manager.submit_job(
                script_content=job_script,
                job_name=f"evopt_{job_id}",
                cpus_per_task=self.cores_per_worker,
                output_dir=job_dir,
                memory_gb=self.memory_gb,
                wall_time=self.wall_time,
                qos=self.qos
            )
            
            # Store job info
            self._jobs[slurm_job_id] = (future, output_file, job_dir)
            
            # Start monitoring thread
            self._start_monitor_thread(slurm_job_id, future, output_file, job_dir)
            
            return future
            
        except Exception as e:
            # Release semaphore on error
            self._job_semaphore.release()
            future.set_exception(e)
            return future
    
    def _start_monitor_thread(self, job_id, future, result_path, job_dir):
        """Start a thread to monitor job completion.
        
        Creates and starts a daemon thread that monitors the status of a SLURM job.
        The thread waits for job completion, checks for output files, and sets the
        result or exception on the associated Future object.
        
        Args:
            job_id (int): The SLURM job ID to monitor.
            future (concurrent.futures.Future): The Future associated with the job.
            result_path (str): Path to the file where the job result will be stored.
            job_dir (str): Directory containing job files.
            
        Returns:
            None: This method does not return a value.
            
        Note:
            This is an internal method used by the submit method.
        """

        def monitor_job():
            try:
                # Wait for job to complete
                self.job_manager.wait_for_job(job_id)
                
                # Wait for completion file (may take a moment after job finishes)
                completion_file = os.path.join(job_dir, "COMPLETED")
                max_wait = 30  # seconds
                wait_start = time.time()
                
                while not os.path.exists(completion_file) and time.time() - wait_start < max_wait:
                    time.sleep(1)
                
                # Load result if available
                if os.path.exists(result_path):
                    with open(result_path, 'rb') as f:
                        success, result, error = pickle.load(f)
                        
                    if success:
                        future.set_result(result)
                    else:
                        future.set_exception(RuntimeError(f"Job failed: {error}"))
                else:
                    future.set_exception(RuntimeError(f"Job completed but no result found"))
            except Exception as e:
                future.set_exception(e)
            finally:
                # Release semaphore to allow another job
                self._job_semaphore.release()
                
        thread = threading.Thread(target=monitor_job)
        thread.daemon = True
        thread.start()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor and clean up resources.
        
        This method marks the executor as shutdown, cancels any running SLURM jobs,
        and prevents new jobs from being submitted. If wait is True, it will not
        return until all currently executing jobs have completed.
        
        Args:
            wait: Whether to wait for running jobs to complete before returning.
                Default is True.
                
        Returns:
            None
            
        Example:
            >>> # Shutdown and cancel all running jobs
            >>> executor.shutdown(wait=False)
            >>>
            >>> # Shutdown and wait for all jobs to complete
            >>> executor.shutdown(wait=True)
        """
        
        self._shutdown = True
        
        # Cancel all running jobs
        for job_id, (future, _, _) in list(self._jobs.items()):
            if not future.done():
                self.job_manager.cancel_job(job_id)
                future.cancel()