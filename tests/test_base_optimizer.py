import pytest
import os
import shutil
import numpy as np
from evopt.base_optimizer import BaseOptimizer
from evopt.directory_manager import DirectoryManager

class MockEvaluator:
	"""A mock evaluator class for testing."""
	def __init__(self, return_value=0.0):
		self.return_value = return_value
		self.call_count = 0

	def __call__(self, param_dict):
		self.call_count += 1
		return self.return_value

class DummyOptimizer(BaseOptimizer):
	"""A dummy optimizer class for testing BaseOptimizer."""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setup_opt_called = False
		self.check_termination_called = False
		self.optimize_called = False

	def setup_opt(self, epoch=None):
		self.setup_opt_called = True

	def check_termination(self):
		self.check_termination_called = True
		return True  # Always terminate for testing

	def optimize(self):
		self.optimize_called = True

@pytest.fixture
def setup_test_environment():
	"""Fixture to set up and tear down the test environment."""
	test_dir = "test_base_optimizer_dir"
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)
	yield test_dir  # Provide the directory to the test
	shutil.rmtree(test_dir)  # Clean up after the test

@pytest.fixture
def base_optimizer_instance(setup_test_environment):
	"""Fixture to create a BaseOptimizer instance for testing."""
	params = {
		'param1': (0, 1),
		'param2': (0, 2),
	}
	evaluator = MockEvaluator()
	n_epochs = 10
	batch_size = 5
	base_dir = setup_test_environment
	directory_manager = DirectoryManager(base_dir=base_dir)
	sigma_threshold = 0.01
	rand_seed = 1
	start_epoch = None
	verbose = False
	optimizer = DummyOptimizer(
		parameters=params,
		evaluator=evaluator,
		batch_size=batch_size,
		directory_manager=directory_manager,
		sigma_threshold=sigma_threshold,
		rand_seed=rand_seed,
		start_epoch=start_epoch,
		verbose=verbose,
		n_epochs=n_epochs
	)
	return optimizer

def test_base_optimizer_init(base_optimizer_instance):
	"""Test the initialisation of the BaseOptimizer."""
	optimizer = base_optimizer_instance
	assert optimizer.parameters == {'param1': (0, 1), 'param2': (0, 2)}
	assert isinstance(optimizer.evaluator, MockEvaluator)
	assert optimizer.n_epochs == 10
	assert optimizer.batch_size == 5
	assert isinstance(optimizer.dir_manager, DirectoryManager)
	assert optimizer.sigma_threshold == 0.01
	assert optimizer.rand_seed == 1
	assert optimizer.start_epoch is None
	assert optimizer.verbose == False
	assert optimizer.current_epoch == 0

def test_get_init_sigmas(base_optimizer_instance):
	"""Test the get_init_sigmas property."""
	optimizer = base_optimizer_instance
	expected_sigmas = np.array([0.25, 0.5])
	np.testing.assert_allclose(optimizer.get_init_sigmas, expected_sigmas)

def test_get_norm_bounds(base_optimizer_instance):
	"""Test the get_norm_bounds property."""
	optimizer = base_optimizer_instance
	expected_bounds = [(0, 4), (0, 4)]
	assert optimizer.get_norm_bounds == expected_bounds

def test_get_init_params(base_optimizer_instance):
	"""Test the get_init_params property."""
	optimizer = base_optimizer_instance
	init_params = optimizer.get_init_params
	assert len(init_params) == 2
	assert all(0 <= param <= 4 for param in init_params)

def test_rescale_params(base_optimizer_instance):
	"""Test the rescale_params method."""
	optimizer = base_optimizer_instance
	params = np.array([1.0, 2.0])
	expected_rescaled_params = np.array([0.25, 1.0])
	np.testing.assert_allclose(optimizer.rescale_params(params), expected_rescaled_params)

def test_process_batch(base_optimizer_instance):
	"""Test the process_batch method."""
	optimizer = base_optimizer_instance
	solutions = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
	errors = optimizer.process_batch(solutions)
	assert len(errors) == 2
	assert optimizer.evaluator.call_count == 2

def test_setup_opt_abstract_method(base_optimizer_instance):
	"""Test the setup_opt abstract method."""
	optimizer = base_optimizer_instance
	optimizer.setup_opt()
	assert optimizer.setup_opt_called == True

def test_check_termination_abstract_method(base_optimizer_instance):
	"""Test the check_termination abstract method."""
	optimizer = base_optimizer_instance
	result = optimizer.check_termination()
	assert optimizer.check_termination_called == True
	assert result == True

def test_optimize_abstract_method(base_optimizer_instance):
	"""Test the optimize abstract method."""
	optimizer = base_optimizer_instance
	optimizer.optimize()
	assert optimizer.optimize_called == True
