import pytest
from evopt import optimise
import os
import shutil

def mock_evaluator(param_dict):
    """A simple mock evaluator for testing."""
    return sum(param_dict.values())

@pytest.fixture
def setup_test_environment():
    """Fixture to set up and tear down the test environment."""
    test_dir = "test_evopt_dir"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    yield test_dir  # Provide the directory to the test
    shutil.rmtree(test_dir)  # Clean up after the test

def test_optimise_cmaes(setup_test_environment):
    """Test the optimise function with CMA-ES."""
    params = {
        'param1': (0, 1),
        'param2': (0, 1),
    }
    n_epochs = 2
    batch_size = 5
    base_dir = setup_test_environment
    dir_id = 0  # Ensure consistent directory naming for testing

    optimise(
        params=params,
        evaluator=mock_evaluator,
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimiser='cmaes',
        base_dir=base_dir,
        dir_id=dir_id,
        verbose=False  # Suppress verbose output during testing
    )

    # Assert that the expected directories and files were created
    evolve_dir = os.path.join(base_dir, "evolve_0")
    assert os.path.exists(evolve_dir)
    assert os.path.exists(os.path.join(evolve_dir, "epochs.csv"))
    assert os.path.exists(os.path.join(evolve_dir, "results.csv"))
    assert os.path.exists(os.path.join(evolve_dir, "epochs"))
    assert os.path.exists(os.path.join(evolve_dir, "checkpoints"))
    assert os.path.exists(os.path.join(evolve_dir, "logs"))

def test_optimise_unsupported_optimiser(setup_test_environment):
    """Test that an exception is raised for an unsupported optimiser."""
    params = {
        'param1': (0, 1),
        'param2': (0, 1),
    }
    n_epochs = 2
    batch_size = 5
    base_dir = setup_test_environment
    dir_id = 0

    with pytest.raises(ValueError, match="Unsupported optimizer: invalid"):
        optimise(
            params=params,
            evaluator=mock_evaluator,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimiser='invalid',
            base_dir=base_dir,
            dir_id=dir_id
        )
