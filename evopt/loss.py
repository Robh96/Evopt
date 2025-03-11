"""Multi-objective loss calculation for evolutionary optimization.

This module provides functionality for calculating loss values in optimization scenarios
with multiple objectives and constraints. It supports both hard and soft constraints, 
with configurable weighting between them.

The module implements normalized error calculations to handle objectives with different
scales and provides mechanisms to distinguish between hard constraints (must be satisfied)
and soft objectives (should be optimized).

Example:
    Basic usage with target values and observed results:
    
    >>> from evopt.loss import calc_loss
    >>> 
    >>> # Define target values with constraints
    >>> targets = {
    ...     'weight': 10.0,               # Target weight (default: hard constraint)
    ...     'stress': (0, 50),            # Stress must be between 0 and 50
    ...     'deflection': {'value': 0.5, 'hard': False}  # Soft objective
    ... }
    >>> 
    >>> # Observed values from evaluation
    >>> observed = {
    ...     'weight': 12.5,
    ...     'stress': 30.2,
    ...     'deflection': 0.8
    ... }
    >>> 
    >>> # Calculate loss
    >>> loss = calc_loss(targets, observed, method="mae")
    >>> print(f"Combined loss: {loss.combined_loss}")
    >>> print(f"Observed values: {loss.observed_dict}")
"""

import numpy as np

class CLoss:
    """Loss calculator for multi-objective optimization with constraints.
    
    This class handles the calculation of loss values when optimizing against
    multiple objectives with both hard and soft constraints. It normalizes
    errors across different scales and can weight hard constraints more heavily
    than soft objectives.
    
    Hard constraints are treated as requirements that must be satisfied, while
    soft objectives are goals that should be optimized but can be compromised.
    The final loss combines both types with configurable weighting.
    
    Attributes:
        target_dict: Dictionary of target values with optional constraint settings.
        method: Error calculation method ('mse' or 'mae').
        verbose: Whether to print detailed information during calculation.
        w: Weight factor for hard constraints (between 0 and 1).
        observed_dict: Dictionary of mean observed values after calculation.
        combined_loss: The final weighted loss value after calculation.
        
    Examples:
        Creating a loss calculator and computing loss:
        
        >>> targets = {'stress': (0, 100), 'weight': 50.0}
        >>> observed = {'stress': 80.0, 'weight': 55.2}
        >>> loss_calc = CLoss(targets, method='mae', verbose=True)
        >>> loss_calc.calc_loss(observed)
        >>> print(f"Loss: {loss_calc.combined_loss}")
    """
    
    def __init__(
            self,
            target_dict: dict,
            method="mae",
            verbose: bool = False,
            hard_to_soft_weight: float = 0.9
    ):
        """Initialize the Loss object with target values and settings.
        
        Args:
            target_dict: Dictionary of target values. Each key can have:
                - A simple numeric value (treated as a hard constraint)
                - A tuple of (min, max) representing a range (hard constraint)
                - A dict with 'value' and optional 'hard' keys
            method: Error calculation method, either 'mse' (mean squared error) or
                'mae' (mean absolute error). Default is 'mae'.
            verbose: Whether to print detailed information during calculation.
                Default is False.
            hard_to_soft_weight: Weight factor for balancing hard and soft constraints.
                1.0 means only hard constraints matter, 0.0 means only soft constraints.
                Default is 0.9 (90% weight to hard constraints).
                
        Example:
            >>> # Define targets with mixed constraint types
            >>> targets = {
            ...     'weight': 10.0,  # Hard constraint (default)
            ...     'stress': (0, 50),  # Hard constraint range
            ...     'deflection': {'value': 0.5, 'hard': False}  # Soft objective
            ... }
            >>> loss_calc = CLoss(targets, method='mae', verbose=True)
        """

        self.target_dict = target_dict
        self.method = method.lower() if method.lower() in ["mse", "mae"] else "mse"
        self.verbose = verbose
        self.w = hard_to_soft_weight # weight for hard constraints
        self.observed_dict = {}
        self.combined_loss = None

    def calculate_error(self, target_values, observed_values) -> float:
        """Calculate normalized error between target and observed values.
        
        Computes a normalized error measure that handles different scales automatically.
        The normalization divides the difference by the sum of absolute values plus 1,
        making the error scale-independent. NaN values in observed data are ignored.
        
        Args:
            target_values: List or array of target values.
            observed_values: List or array of observed values.
            
        Returns:
            float: Normalized error value (MAE or MSE based on the configured method).
            
        Note:
            Returns NaN if all observed values are NaN.
            
        Example:
            >>> loss_calc = CLoss({"dummy": 0})
            >>> error = loss_calc.calculate_error([10, 20, 30], [12, 18, 35])
            >>> print(f"Error: {error:.4f}")  # Output depends on method
        """

        target_values = np.array(target_values)
        observed_values = np.array(observed_values)
        mask = ~np.isnan(observed_values)
        if np.sum(mask) == 0:
            return np.nan
        target_values = target_values[mask]
        observed_values = observed_values[mask]
        err = ((observed_values - target_values) / (np.abs(target_values) + np.abs(observed_values) + 1))
        
        if self.method == "mae":
            return np.nanmean(np.abs(err))
        else:
            return np.nanmean(err ** 2)

    def constraint_satisfied(self, key: str, observed_values: list) -> bool:
        """Check if observed values satisfy the constraint for a given key.
        
        Determines whether observed values meet the constraint defined in target_dict.
        For single values, a 5% tolerance is applied unless the constraint is defined
        as a range using a tuple.
        
        Args:
            key: The key in target_dict to check.
            observed_values: List of observed values to check against the constraint.
            
        Returns:
            bool: True if the constraint is satisfied (at least 50% of values within bounds),
                False otherwise.
                
        Note:
            Returns False if observed_values is empty or contains only None/NaN.
            
        Example:
            >>> targets = {'stress': (0, 100)}  # Stress must be between 0 and 100
            >>> loss_calc = CLoss(targets)
            >>> satisfied = loss_calc.constraint_satisfied('stress', [50, 80, 120])
            >>> print(f"Constraint satisfied: {satisfied}")  # True (2/3 within bounds)
        """

        constraint_info = self.target_dict[key]
        if isinstance(constraint_info, dict):
            target_val = constraint_info.get("value", constraint_info)
        else:
            target_val = constraint_info
        observed_values = [v for v in observed_values if v is not None and not np.isnan(v)]
        if not observed_values:
            if self.verbose:
                print(f"Warning: {key} has no observed values.")
            return False
        
        if isinstance(target_val, tuple):
            min_val, max_val = target_val
        else:
            tolerance = 0.05 * abs(target_val) if target_val != 0 else 5e-2
            min_val, max_val = target_val - tolerance, target_val + tolerance

        outside_count = sum(1 for v in observed_values if not (min_val <= v <= max_val))
        if self.verbose:
            print(f"{key}: {100 * outside_count / len(observed_values):.0f}% of values outside [{min_val:.2e}, {max_val:.2e}]")
        return outside_count / len(observed_values) <= 0.5

    def calc_loss(self, observed_dict: dict) -> float:
        """Calculate the combined loss across all constraints and objectives.
        
        Processes the observed values for each key in target_dict, computing
        individual errors and determining if constraints are satisfied. Hard constraints
        and soft objectives are weighted according to the hard_to_soft_weight parameter.
        
        Args:
            observed_dict: Dictionary of observed values with the same keys as target_dict.
                Values can be single numbers, lists, or arrays.
                
        Returns:
            float: The combined loss value is stored in self.combined_loss but not returned.
            
        Raises:
            KeyError: If a key in target_dict is not found in observed_dict.
            
        Note:
            After calling this method, access the results via self.combined_loss
            and self.observed_dict properties.
            
        Example:
            >>> targets = {
            ...     'weight': 10.0,
            ...     'stress': {'value': 50.0, 'hard': True},
            ...     'deflection': {'value': 0.5, 'hard': False}
            ... }
            >>> observed = {
            ...     'weight': 12.5,
            ...     'stress': 80.0,
            ...     'deflection': 0.8
            ... }
            >>> loss_calc = CLoss(targets)
            >>> loss_calc.calc_loss(observed)
            >>> print(f"Combined loss: {loss_calc.combined_loss}")
        """

        observed_dict = {k: self._convert_to_native(v) for k, v in observed_dict.items()}
        hard_losses = []
        soft_losses = []

        for key, constraint_info in self.target_dict.items():
            if key in observed_dict:
                observed_val = observed_dict[key]
                is_hard = True

                if isinstance(constraint_info, dict):
                    is_hard = constraint_info.get("hard", True)
                    target_val = constraint_info.get("value", constraint_info)
                else:
                    target_val = constraint_info
                if isinstance(target_val, tuple):
                    target_val = np.mean(target_val)
                
                loss = self.calculate_error([target_val] * len(observed_val), observed_val)
                
                constraint_met = self.constraint_satisfied(key, observed_val)
                if not constraint_met:
                    hard_losses.append(loss) if is_hard else soft_losses.append(loss)

                if self.verbose:
                    print(f"{key}: {np.nanmean(observed_val):.2f} | loss: {loss:.2e} | Hard: {is_hard} | Constraint met: {constraint_met}")
            else:
                raise KeyError(f"Observed data missing for key: {key}")
                
        hard_loss = np.nanmean(hard_losses) if hard_losses else 0.0
        soft_loss = np.nanmean(soft_losses) if soft_losses else 0.0
        self.combined_loss = self.w * hard_loss + (1 - self.w) * soft_loss if hard_losses else (1 - self.w) * soft_loss # more stable condition
        self.observed_dict = {key: np.nanmean(observed_dict[key]) for key in observed_dict}

    def _convert_to_native(self, value) -> list[float]:
        """Convert various input types to lists of native Python floats.
        
        Helper method to standardize different input formats to lists of
        floats for consistent processing.
        
        Args:
            value: Input value of various possible types (scalar, list, array, dict).
            
        Returns:
            List[float]: List of float values, or the original value if conversion
                is not applicable.
                
        Note:
            This is an internal helper method not meant for direct use.
        """

        if isinstance(value, (np.float64, float, int)):
            return [float(value)]
        elif isinstance(value, (list, np.ndarray, tuple)):
            return [float(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._convert_to_native(v) for k, v in value.items()}
        return value

def calc_loss(
        target_dict: dict,
        observed_dict: dict,
        method: str = "mse",
        hard_to_soft_weight: float = 0.9,
        verbose: bool = False
    ) -> CLoss:
    """Convenience function to calculate loss for multi-objective optimization.
    
    Creates a CLoss object with the specified parameters, calculates the loss,
    and returns the configured loss calculator for further access to results.
    
    Args:
        target_dict: Dictionary of target values and constraint specifications.
            See CLoss class documentation for format details.
        observed_dict: Dictionary of observed values to evaluate.
        method: Error calculation method ('mse' or 'mae'). Default is 'mse'.
        hard_to_soft_weight: Weight for balancing hard vs. soft constraints (0.0-1.0).
            Default is 0.9 (90% weight to hard constraints).
        verbose: Whether to print detailed calculation information. Default is False.
            
    Returns:
        CLoss: Configured loss calculator object with results accessible via
            the combined_loss and observed_dict properties.
            
    Raises:
        KeyError: If a key in target_dict is not found in observed_dict.
        
    Example:
        >>> targets = {'weight': 10.0, 'stress': (0, 100)}
        >>> observed = {'weight': 12.5, 'stress': 80.0}
        >>> loss = calc_loss(targets, observed, method='mae', verbose=True)
        >>> print(f"Combined loss: {loss.combined_loss}")
        >>> print(f"Observed values: {loss.observed_dict}")
    """
    
    loss_function = CLoss(
        target_dict=target_dict,
        method=method,
        verbose=verbose,
        hard_to_soft_weight=hard_to_soft_weight
    )
    loss_function.calc_loss(observed_dict)
    return loss_function