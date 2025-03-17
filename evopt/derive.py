from pysr import PySRRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import textwrap

class Derive:
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
            #additional_operators:dict=None
        ):
        self.evolve_dir_path = evolve_dir_path
        self.target_variable = target_variable
        self.parameters = parameters
        self.save_dir = save_dir if save_dir else os.path.join(self.evolve_dir_path, "equations")
        self.binary_operators = binary_operators if binary_operators else ["+", "-", "*", "/", "^"]
        self.unary_operators = unary_operators if unary_operators else ["sin", "cos", "exp", "log"]
        self.n_iterations = n_iterations
        self.population_size = population_size
        # self.additional_operators = additional_operators
        self.results_csv_path = os.path.join(self.evolve_dir_path, "results.csv")
        self.y_pred = None
        self.best_equation = None
        
        # # create dictionary of additional operators
        # if self.additional_operators:
        #     self.additional_operators = {k: lambda x: eval(v) for k, v in self.additional_operators.items()}
        if not os.path.exists(self.results_csv_path):
            raise FileNotFoundError(f"File not found: {self.results_csv_path}")
        os.makedirs(self.save_dir, exist_ok=True)

        data = pd.read_csv(self.results_csv_path)
        self.data = data.dropna()
        self.y_target = self.data[self.target_variable]
        self.X_parameters = self.data[self.parameters]

    def _get_id(self) -> str:
        files = [f for f in os.listdir(self.save_dir) if f.startswith("equations_")]
        existing_ids = sorted([int(f.split("_")[-1]) for f in files if f.split("_")[-1].isdigit()])
        
        # Find the smallest missing ID
        if not existing_ids:
            return "equations_0"
        return f"equations_{next((i for i in range(max(existing_ids) + 2) if i not in existing_ids), 0)}"

    def fit(self):

        constraints = {
            "^": (-1, 1)
            }
        nested_constraints = {
            "sin": {"cos": 0},
            "cos": {"sin": 0},
            "exp": {"log": 0},
            "log": {"exp": 0},
            "sin": {"sin": 0},
            "cos": {"cos": 0},
            "exp": {"exp": 0},
            "log": {"log": 0}
        }
        self.model = PySRRegressor(
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            parsimony=0.0032,
            turbo=True,
            niterations=self.n_iterations,
            population_size=self.population_size,
            # extra_sympy_mappings=self.additional_operators,
            output_directory=self.save_dir,
            run_id=self._get_id(),
            constraints=constraints,
            nested_constraints=nested_constraints
            )
        self.model.fit(X=self.X_parameters, y=self.y_target)
        self.best_equation = self.model.sympy()
    
    def predict(self, index:int=None):
        # if index is None, the best_equation is selected for predictions.
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
        """
        Plot the parity plot of the target variable and the predicted variable.
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

        title = title if title else f"parity plot of {self.target_variable}"
        fig, ax = plt.subplots()
        ax.scatter(
            self.y_target,
            self.y_pred,
            marker="o",
            c=point_colour,
            s=8,
            alpha=alpha
        )
        ax.set_xlabel(self.target_variable)
        ax.set_ylabel(wrapped_label)
        ax.set_title(title)

        min_val = np.min([self.y_target.to_numpy().min(), self.y_pred.to_numpy().min()])
        max_val = np.max([self.y_target.to_numpy().max(), self.y_pred.to_numpy().max()])

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        file_name = f"{self.target_variable} parity_plot.{save_ext}"

        if save_figures:
            plt.savefig(os.path.join(save_dir, file_name))
        
        if show:
            plt.show()
        plt.close()
        return ax
