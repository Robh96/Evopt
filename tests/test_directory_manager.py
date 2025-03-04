import pytest
import os
import shutil
import tempfile
from evopt.directory_manager import DirectoryManager

@pytest.fixture
def setup_test_environment():
    """Fixture to set up and tear down the test environment."""
    base_dir = tempfile.mkdtemp()  # Create a temporary directory
    test_dir = os.path.join(base_dir, "test_directory_manager_dir") # create test_dir inside the temp directory
    
    # Remove any existing test directory
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
            
    os.makedirs(test_dir)
    yield test_dir  # Provide the directory to the test
    
    # Teardown: Remove the base directory after the test has run
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

def test_directory_manager_init(setup_test_environment):
    """Test the initialisation of the DirectoryManager."""
    base_dir = setup_test_environment
    dir_id = 0
    directory_manager = DirectoryManager(base_dir=base_dir, dir_id=dir_id)

    assert directory_manager.base_dir == base_dir
    assert directory_manager.dir_id == dir_id
    assert directory_manager.evolve_dir == os.path.join(base_dir, "evolve_0")
    assert os.path.exists(directory_manager.evolve_dir)
    assert os.path.exists(directory_manager.epochs_dir)
    assert os.path.exists(directory_manager.checkpoint_dir)

def test_get_dir_id_no_existing_directories(setup_test_environment):
    """Test get_dir_id when no existing directories are present."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir, dir_id=0)
    evolve_dir = os.path.join(base_dir, "evolve_0")
    assert directory_manager.evolve_dir == os.path.join(base_dir, "evolve_0")

def test_get_dir_id_existing_directories(setup_test_environment):
    """Test get_dir_id with existing directories."""
    base_dir = setup_test_environment
    os.makedirs(os.path.join(base_dir, "evolve_0"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "evolve_1"), exist_ok=True)
    directory_manager = DirectoryManager(base_dir=base_dir, dir_id=2)
    assert directory_manager.evolve_dir == os.path.join(base_dir, "evolve_2")

def test_get_dir_id_specified_dir_id(setup_test_environment):
    """Test get_dir_id with a specified dir_id."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir, dir_id=5)
    assert directory_manager.get_dir_id(dir_id=5) == 5
    assert directory_manager.dir_id == 5

def test_working_directory(setup_test_environment):
    """Test the working_directory context manager."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    original_cwd = os.getcwd()
    
    with directory_manager.working_directory(directory_manager.evolve_dir):
        # Check if the working directory has changed within the context
        assert os.getcwd() != original_cwd
        assert os.path.abspath(os.getcwd()) == os.path.abspath(directory_manager.evolve_dir)
        
    # Check if the working directory has been restored after the context
    assert os.getcwd() == original_cwd

def test_create_epoch_folder(setup_test_environment):
    """Test the create_epoch_folder method."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    epoch_folder = directory_manager.create_epoch_folder(epoch=0)
    assert os.path.exists(epoch_folder)
    assert os.path.basename(epoch_folder) == "epoch0000"

def test_create_solution_folder(setup_test_environment):
    """Test the create_solution_folder method."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    solution_folder = directory_manager.create_solution_folder(epoch=0, solution=0)
    assert os.path.exists(solution_folder)
    assert os.path.basename(solution_folder) == "solution0000"
    assert os.path.basename(os.path.dirname(solution_folder)) == "epoch0000"

def test_get_checkpoint_filepath(setup_test_environment):
    """Test the get_checkpoint_filepath method."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    filepath = directory_manager.get_checkpoint_filepath(epoch=0)
    assert filepath == os.path.join(directory_manager.checkpoint_dir, "checkpoint_epoch0000.pkl")

def test_save_and_load_checkpoint(setup_test_environment):
    """Test the save_checkpoint and load_checkpoint methods."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    data = {"test": "data"}
    epoch = 1
    directory_manager.save_checkpoint(data=data, epoch=epoch)
    loaded_data = directory_manager.load_checkpoint(epoch=epoch)
    assert loaded_data == data

def test_load_latest_checkpoint(setup_test_environment):
    """Test loading the latest checkpoint."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    data1 = {"test": "data1"}
    data2 = {"test": "data2"}
    directory_manager.save_checkpoint(data=data1, epoch=1)
    directory_manager.save_checkpoint(data=data2, epoch=2)
    loaded_data = directory_manager.load_checkpoint()
    assert loaded_data == data2

def test_load_checkpoint_no_checkpoints(setup_test_environment):
    """Test loading a checkpoint when no checkpoints exist."""
    base_dir = setup_test_environment
    directory_manager = DirectoryManager(base_dir=base_dir)
    loaded_data = directory_manager.load_checkpoint()
    assert loaded_data is None
