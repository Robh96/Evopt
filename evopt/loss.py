import numpy as np

class CLoss:
    def __init__(self, target_dict: dict, method="mse", verbose: bool = True, hard_to_soft_weight: float = 0.9):
        """Initializes the Loss object with target values."""
        self.target_dict = target_dict
        # choose between normalised mse or normalised mae
        self.method = method.lower() if method.lower() in ["mse", "mae"] else "mse"
        self.verbose = verbose
        self.w = hard_to_soft_weight # weight for hard constraints
        self.observed_dict = {}
        self.combined_loss = None

    def calculate_error(self, target_values, observed_values):
        """Calculates error between lists/arrays, ignoring NaNs."""
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

    def constraint_satisfied(self, key, observed_values):
        """Checks if hard constraints are satisfied with 5% tolerance."""
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
        """Calculates the loss across all keys with dynamic weighting."""
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

    def _convert_to_native(self, value):
        if isinstance(value, (np.float64, float, int)):
            return [float(value)]
        elif isinstance(value, (list, np.ndarray, tuple)):
            return [float(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._convert_to_native(v) for k, v in value.items()}
        return value

def calc_loss(target_dict: dict, observed_dict: dict, method="mse", verbose: bool = True, hard_to_soft_weight: float = 0.8):
    loss_function = CLoss(target_dict=target_dict, method=method, verbose=verbose, hard_to_soft_weight=hard_to_soft_weight)
    loss_function.calc_loss(observed_dict)
    return loss_function