import os
import time
import pickle
import concurrent.futures
import tempfile
import threading
from .utils import SlurmJobManager

class SlurmExecutor(concurrent.futures.Executor):
    """Executor that submits tasks as SLURM jobs"""
    
    def __init__(
            self,
            max_workers=1,
            cores_per_worker=1,
            memory_gb=4, 
            wall_time="1:00:00",
            qos=None
        ):
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
        """Submit a function for execution as a SLURM job"""
        if self._shutdown:
            raise RuntimeError("Executor is shutdown")
            
        # Wait for a job slot to become available
        self._job_semaphore.acquire()
        
        # Create future
        future = concurrent.futures.Future()
        
        # Extract working directory from args if it's the BaseOptimiser._evaluate_solution_worker
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
        """Start a thread to monitor job completion"""
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
    
    def shutdown(self, wait=True):
        """Shutdown the executor"""
        self._shutdown = True
        
        # Cancel all running jobs
        for job_id, (future, _, _) in list(self._jobs.items()):
            if not future.done():
                self.job_manager.cancel_job(job_id)
                future.cancel()