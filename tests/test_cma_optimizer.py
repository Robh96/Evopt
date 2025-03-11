import pytest
import os
import shutil
import numpy as np
import cma
from evopt.cma_optimizer import CmaesOptimizer
from evopt.directory_manager import DirectoryManager

class MockEvaluator:
    """A mock evaluator class for testing."""
    def __init__(self, return_value=0.0):
        self.return_value = return_value
        self.call_count = 0

    def __call__(self, param_dict):
        self.call_count += 1
        return self.return_value

@pytest.fixture
def setup_test_environment():
    """Fixture to set up and tear down the test environment."""
    test_dir = "test_cma_optimizer_dir"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    yield test_dir  # Provide the directory to the test
    shutil.rmtree(test_dir)  # Clean up after the test

@pytest.fixture
def cma_optimizer_instance(setup_test_environment):
    """Fixture to create a CmaesOptimizer instance for testing."""
    params = {
        'param1': (0, 1),
        'param2': (0, 2),
    }
    evaluator = MockEvaluator()
    n_epochs = 2
    batch_size = 5
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    optimizer = CmaesOptimizer(
        parameters=params,
        evaluator=evaluator,
        n_epochs=n_epochs,
        batch_size=batch_size,
        directory_manager=directory_manager,
        sigma_threshold=0.01,
        rand_seed=1,
        verbose=False
    )
    return optimizer

def test_cma_optimizer_init(cma_optimizer_instance):
    """Test the initialisation of the CmaesOptimizer."""
    optimizer = cma_optimizer_instance
    assert optimizer.parameters == {'param1': (0, 1), 'param2': (0, 2)}
    assert isinstance(optimizer.evaluator, MockEvaluator)
    assert optimizer.n_epochs == 2
    assert optimizer.batch_size == 5
    assert isinstance(optimizer.dir_manager, DirectoryManager)
    assert optimizer.sigma_threshold == 0.01
    assert optimizer.rand_seed == 1
    assert optimizer.verbose == False
    assert optimizer.current_epoch == 0

def test_setup_opt_new_run(cma_optimizer_instance):
    """Test the setup_opt method for a new CMA-ES run."""
    optimizer = cma_optimizer_instance
    optimizer.setup_opt()
    assert isinstance(optimizer.es, cma.CMAEvolutionStrategy)
    assert optimizer.current_epoch == 0

def test_setup_opt_resume_run(cma_optimizer_instance):
    """Test the setup_opt method for resuming a CMA-ES run from a checkpoint."""
    optimizer = cma_optimizer_instance
    # Create a dummy checkpoint file
    es = cma.CMAEvolutionStrategy([0.5, 1.0], 0.1)
    optimizer.dir_manager.save_checkpoint(es, epoch=0)
    
    optimizer.setup_opt(epoch=0)
    assert isinstance(optimizer.es, cma.CMAEvolutionStrategy)
    assert optimizer.current_epoch == 0

def test_check_termination(cma_optimizer_instance):
    """Test the check_termination method."""
    optimizer = cma_optimizer_instance
    optimizer.setup_opt()  # Initialise es
    # Mock the sigma values to be below the threshold
    optimizer.es.sigma = 0.001
    optimizer.es.C = np.eye(2)  # Ensure C is positive definite
    assert optimizer.check_termination() == True

    # Mock the sigma values to be above the threshold
    optimizer.es.sigma = 1.0
    assert optimizer.check_termination() == False

def test_optimize(cma_optimizer_instance, mocker):
    """Test the optimize method."""
    optimizer = cma_optimizer_instance
    
    # Mock the check_termination method to return True after one iteration
    mocker.patch.object(optimizer, 'check_termination', side_effect=[False, True])
    
    optimizer.optimize()
    
    assert optimizer.check_termination.call_count == 2
    assert optimizer.current_epoch == 1

def test_optimize_max_epochs(cma_optimizer_instance, mocker):
    """Test the optimize method when max epochs is reached."""
    optimizer = cma_optimizer_instance
    
    # Ensure n_epochs is not None (epochs 0 and 1)
    optimizer.n_epochs = 2
    
    # Mock the check_termination method to always return False (never meets sigma threshold)
    mocker.patch.object(optimizer, 'check_termination', side_effect=[False, False, True])
    
    optimizer.optimize()
    
    assert optimizer.check_termination.call_count == optimizer.n_epochs + 1
    assert optimizer.current_epoch == optimizer.n_epochs

def test_optimize_sigma_threshold(cma_optimizer_instance, mocker):
    """Test the optimize method when sigma threshold is met."""
    optimizer = cma_optimizer_instance
    
    # Ensure n_epochs is None, so it terminates based on sigma threshold
    optimizer.n_epochs = None
    
    # Mock the check_termination method to return True after one iteration
    mocker.patch.object(optimizer, 'check_termination', side_effect=[False, True])
    
    optimizer.optimize()
    
    assert optimizer.check_termination.call_count == 2
    assert optimizer.current_epoch == 1
