import pytest
import os
import shutil
import numpy as np
import cma
from evopt.cma_optimiser import CmaesOptimiser
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
    test_dir = "test_cma_optimiser_dir"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    yield test_dir  # Provide the directory to the test
    shutil.rmtree(test_dir)  # Clean up after the test

@pytest.fixture
def cma_optimiser_instance(setup_test_environment):
    """Fixture to create a CmaesOptimiser instance for testing."""
    params = {
        'param1': (0, 1),
        'param2': (0, 2),
    }
    evaluator = MockEvaluator()
    n_epochs = 2
    batch_size = 5
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    optimiser = CmaesOptimiser(
        parameters=params,
        evaluator=evaluator,
        n_epochs=n_epochs,
        batch_size=batch_size,
        directory_manager=directory_manager,
        sigma_threshold=0.01,
        rand_seed=1,
        verbose=False
    )
    return optimiser

def test_cma_optimiser_init(cma_optimiser_instance):
    """Test the initialisation of the CmaesOptimiser."""
    optimiser = cma_optimiser_instance
    assert optimiser.parameters == {'param1': (0, 1), 'param2': (0, 2)}
    assert isinstance(optimiser.evaluator, MockEvaluator)
    assert optimiser.n_epochs == 2
    assert optimiser.batch_size == 5
    assert isinstance(optimiser.dir_manager, DirectoryManager)
    assert optimiser.sigma_threshold == 0.01
    assert optimiser.rand_seed == 1
    assert optimiser.verbose == False
    assert optimiser.current_epoch == 0

def test_setup_opt_new_run(cma_optimiser_instance):
    """Test the setup_opt method for a new CMA-ES run."""
    optimiser = cma_optimiser_instance
    optimiser.setup_opt()
    assert isinstance(optimiser.es, cma.CMAEvolutionStrategy)
    assert optimiser.current_epoch == 0

def test_setup_opt_resume_run(cma_optimiser_instance):
    """Test the setup_opt method for resuming a CMA-ES run from a checkpoint."""
    optimiser = cma_optimiser_instance
    # Create a dummy checkpoint file
    es = cma.CMAEvolutionStrategy([0.5, 1.0], 0.1)
    optimiser.dir_manager.save_checkpoint(es, epoch=0)
    
    optimiser.setup_opt(epoch=0)
    assert isinstance(optimiser.es, cma.CMAEvolutionStrategy)
    assert optimiser.current_epoch == 0

def test_check_termination(cma_optimiser_instance):
    """Test the check_termination method."""
    optimiser = cma_optimiser_instance
    optimiser.setup_opt()  # Initialise es
    # Mock the sigma values to be below the threshold
    optimiser.es.sigma = 0.001
    optimiser.es.C = np.eye(2)  # Ensure C is positive definite
    assert optimiser.check_termination() == True

    # Mock the sigma values to be above the threshold
    optimiser.es.sigma = 1.0
    assert optimiser.check_termination() == False

def test_optimise(cma_optimiser_instance, mocker):
    """Test the optimise method."""
    optimiser = cma_optimiser_instance
    
    # Mock the check_termination method to return True after one iteration
    mocker.patch.object(optimiser, 'check_termination', side_effect=[False, True])
    
    optimiser.optimise()
    
    assert optimiser.check_termination.call_count == 2
    assert optimiser.current_epoch == 1

def test_optimise_max_epochs(cma_optimiser_instance, mocker):
    """Test the optimise method when max epochs is reached."""
    optimiser = cma_optimiser_instance
    
    # Ensure n_epochs is not None (epochs 0 and 1)
    optimiser.n_epochs = 2
    
    # Mock the check_termination method to always return False (never meets sigma threshold)
    mocker.patch.object(optimiser, 'check_termination', side_effect=[False, False, True])
    
    optimiser.optimise()
    
    assert optimiser.check_termination.call_count == optimiser.n_epochs + 1
    assert optimiser.current_epoch == optimiser.n_epochs

def test_optimise_sigma_threshold(cma_optimiser_instance, mocker):
    """Test the optimise method when sigma threshold is met."""
    optimiser = cma_optimiser_instance
    
    # Ensure n_epochs is None, so it terminates based on sigma threshold
    optimiser.n_epochs = None
    
    # Mock the check_termination method to return True after one iteration
    mocker.patch.object(optimiser, 'check_termination', side_effect=[False, True])
    
    optimiser.optimise()
    
    assert optimiser.check_termination.call_count == 2
    assert optimiser.current_epoch == 1
