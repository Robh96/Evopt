"""Symbolic regression utilities for equation discovery from data.

This module provides the Derive class for performing symbolic regression using PySR.
It enables finding mathematical expressions that best fit a given dataset.
"""

from pysr import PySRRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import textwrap

class Derive:
    """Symbolic regression model for equation discovery from data.
    
    This class provides methods to discover mathematical equations that
    describe the relationship between input parameters and a target variable
    using symbolic regression with the PySR library.
    
    Attributes:
        evolve_dir_path (str): Path to the directory containing results data.
        target_variable (str): The variable to be predicted.
        parameters (list[str]): Input parameters to use for prediction.
        save_dir (str): Directory to save equations and output.
        binary_operators (list): Binary operators for symbolic regression.
        unary_operators (list): Unary operators for symbolic regression.
        n_iterations (int): Number of iterations for regression.
        population_size (int): Population size for genetic algorithm.
        max_size (int): Maximum size of generated equations.
        results_csv_path (str): Path to the results CSV file.
        y_pred (DataFrame): Predicted values from the model.
        best_equation (sympy.Expr): Best equation found by symbolic regression.
        sympymappings (dict): Custom operator mappings for SymPy.
    """

    def __init__(
            self,
            evolve_dir_path:str,
            target_variable:str,
            parameters:list[str],
            save_dir:str=None,
            binary_operators:str=None,
            unary_operators:str=None,
            n_iterations:int=100,
            population_size:int=32,
            max_size:int=20,
            #additional_operators:dict=None
        ):
        """Initialize the Derive class for symbolic regression.
        
        Args:
            evolve_dir_path (str): Path to directory containing the results.csv file.
            target_variable (str): Target variable to predict.
            parameters (list[str]): List of parameter names to use as predictors.
            save_dir (str, optional): Directory to save equations. If None, uses
                'equations' subdirectory in evolve_dir_path. Defaults to None.
            binary_operators (list, optional): Binary operators for symbolic regression.
                Defaults to ["+", "-", "*", "/", "^"].
            unary_operators (list, optional): Unary operators for symbolic regression.
                Can include custom operators in format "inv(x)=1/x". 
                Defaults to ["sin", "exp", "log"].
            n_iterations (int, optional): Number of iterations. Defaults to 100.
            population_size (int, optional): Population size. Defaults to 32.
            max_size (int, optional): Maximum size of equations. Defaults to 20.
        
        Raises:
            FileNotFoundError: If results.csv file doesn't exist in evolve_dir_path.
            
        Example:
            >>> derive_model = Derive(
            ...     evolve_dir_path="path/to/data", 
            ...     target_variable="density",
            ...     parameters=["temperature", "pressure"]
            ... )
        """

        self.evolve_dir_path = evolve_dir_path
        self.target_variable = target_variable
        self.parameters = parameters
        self.save_dir = save_dir if save_dir else os.path.join(self.evolve_dir_path, "equations")
        self.binary_operators = binary_operators if binary_operators else ["+", "-", "*", "/", "^"]
        self.unary_operators = unary_operators if unary_operators else ["sin", "exp", "log"]
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.max_size = max_size
        # self.additional_operators = additional_operators
        self.results_csv_path = os.path.join(self.evolve_dir_path, "results.csv")
        self.y_pred = None
        self.best_equation = None
        
        # parse extra operators
        try:
            extra_unary_operators = {op.split("(")[0]: op.split("=")[1] for op in self.unary_operators if "=" in op}
            self.sympymappings = {k: lambda x: eval(v) for k, v in extra_unary_operators.items()}
        except:
            self.sympymappings = None
            print("Error in parsing extra unary operators. Ensure same style as 'inv(x)=1/x'.")

        if not os.path.exists(self.results_csv_path):
            raise FileNotFoundError(f"File not found: {self.results_csv_path}")
        os.makedirs(self.save_dir, exist_ok=True)

        data = pd.read_csv(self.results_csv_path)
        self.data = data.dropna()
        self.y_target = self.data[self.target_variable]
        self.X_parameters = self.data[self.parameters]

    def _get_id(self) -> str:
        """Generate a unique ID for the current regression run.
        
        This method finds the smallest missing ID number from existing equation files
        to ensure unique identification of each symbolic regression run.
        
        Returns:
            str: A unique ID string in the format "equations_X" where X is a number.
            
        Example:
            >>> model._get_id()
            'equations_3'
        """

        files = [f for f in os.listdir(self.save_dir) if f.startswith("equations_")]
        existing_ids = sorted([int(f.split("_")[-1]) for f in files if f.split("_")[-1].isdigit()])
        
        # Find the smallest missing ID
        if not existing_ids:
            return "equations_0"
        return f"equations_{next((i for i in range(max(existing_ids) + 2) if i not in existing_ids), 0)}"

    def fit(self):
        """Fit symbolic regression model to discover equations.
        
        This method configures and runs the PySR symbolic regression algorithm
        to discover mathematical relationships between parameters and target variable.
        It sets constraints on operations and stores the best equation found.
        
        Returns:
            None: Updates self.model and self.best_equation attributes.
            
        Example:
            >>> model = Derive(evolve_dir_path="data", target_variable="y", parameters=["x1", "x2"])
            >>> model.fit()
            >>> print(model.best_equation)
            x1 + 2.5*x2
        """

        constraints = {
            "pow": (9, 1),
            "^": (9, 1)
            }
        nested_constraints = {    
            "exp": {"log": 0},
            "log": {"exp": 0},
            "sin": {"sin": 0},
            "exp": {"exp": 0},
            "log": {"log": 0}
        }
        self.model = PySRRegressor(
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            turbo=True,
            niterations=self.n_iterations,
            population_size=self.population_size,
            # extra_sympy_mappings=self.additional_operators,
            output_directory=self.save_dir,
            run_id=self._get_id(),
            constraints=constraints,
            extra_sympy_mappings=self.sympymappings,
            nested_constraints=nested_constraints,
            adaptive_parsimony_scaling=1000,
            maxsize=self.max_size,
            )
        self.model.fit(X=self.X_parameters, y=self.y_target)
        self.best_equation = self.model.sympy()
    
    def predict(self, index:int=None):
        """Generate predictions using the discovered equation.
        
        Args:
            index (int, optional): Index of the equation to use for prediction.
                If None, uses the best equation. Defaults to None.
        
        Returns:
            None: Updates self.y_pred attribute with prediction results.
            
        Example:
            >>> model.fit()
            >>> model.predict()
            >>> print(model.y_pred.head())
        """

        if self.best_equation is None:
            self.fit()
        y_pred = self.model.predict(self.X_parameters, index=index)
        self.y_pred = pd.DataFrame(y_pred, columns=[self.best_equation])

    def parity_plot(
            self,
            point_colour:str="black",
            alpha:float=0.5,
            title:str=None,
            save_figures:bool=True,
            show:bool=True,
            save_ext:str=".png",
            save_dir:str=None
            ):
        """Plot the parity plot of the target variable and the predicted variable.
        
        This function creates a parity plot comparing actual values with predictions
        from the symbolic regression model, showing how well the discovered equation
        fits the data.
        
        Args:
            point_colour (str, optional): Color of scatter points. Defaults to "black".
            alpha (float, optional): Transparency of points. Defaults to 0.5.
            title (str, optional): Plot title. If None, uses default title. Defaults to None.
            save_figures (bool, optional): Whether to save figure to disk. Defaults to True.
            show (bool, optional): Whether to display the figure. Defaults to True.
            save_ext (str, optional): File extension for saved figure. Defaults to ".png".
            save_dir (str, optional): Directory to save figures. If None, uses
                'figures' subdirectory in evolve_dir_path. Defaults to None.
                
        Returns:
            matplotlib.axes.Axes: The axis object containing the plot.
            
        Raises:
            ValueError: If save_ext is not one of 'png', 'jpg', 'jpeg', 'pdf', or 'svg'.
            
        Example:
            >>> model.fit()
            >>> model.predict()
            >>> model.parity_plot(
            ...     point_colour="blue", 
            ...     alpha=0.7, 
            ...     title="Model Performance",
            ...     save_figures=True
            ... )
        """

        save_dir = save_dir if save_dir else os.path.join(self.evolve_dir_path, "figures")
        save_ext = save_ext.strip(".") if save_ext else "png"
        os.makedirs(save_dir, exist_ok=True)
        if save_ext not in ["png", "jpg", "jpeg", "pdf", "svg"]:
            raise ValueError("Invalid save_ext. Must be one of 'png', 'jpg', 'jpeg', 'pdf', or 'svg'.")
        
        if self.y_pred is None:
            self.predict()

        def format_number(match):
            num = float(match.group(0))
            return f"{num:.3g}"

        formatted_label = re.sub(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", format_number, str(self.best_equation))
        spaced_label = re.sub(r"([()])", r" \1 ", formatted_label)
        wrapped_label = "\n".join(textwrap.wrap(spaced_label, width=40))

        min_val = np.min([self.y_target.to_numpy().min(), self.y_pred.to_numpy().min()])
        max_val = np.max([self.y_target.to_numpy().max(), self.y_pred.to_numpy().max()])

        parity_line = np.linspace(min_val, max_val, 100)

        title = title if title else f"parity plot of {self.target_variable}"
        fig, ax = plt.subplots()
        ax.plot(parity_line, parity_line, linestyle="--", color="red", label="parity line")
        ax.scatter(
            self.y_target,
            self.y_pred,
            marker="o",
            c=point_colour,
            s=8,
            alpha=alpha,
        )
        ax.set_xlabel(self.target_variable)
        ax.set_ylabel(wrapped_label)
        ax.set_title(title)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        file_name = f"{self.target_variable} parity_plot.{save_ext}"

        if save_figures:
            plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
        
        if show:
            plt.show()
        plt.close()
        return ax
