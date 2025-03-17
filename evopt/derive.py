from pysr import PySRRegressor
import pandas as pd
import os

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
        
        # # create dictionary of additional operators
        # if self.additional_operators:
        #     self.additional_operators = {k: lambda x: eval(v) for k, v in self.additional_operators.items()}

    def _get_id(self) -> str:
        files = [f for f in os.listdir(self.save_dir) if f.startswith("equations_")]
        existing_ids = sorted([int(f.split("_")[-1]) for f in files if f.split("_")[-1].isdigit()])
        
        # Find the smallest missing ID
        if not existing_ids:
            return "0"
        return f"{next((i for i in range(max(existing_ids) + 2) if i not in existing_ids), 0)}"

    def run_pysr(self):
        if not os.path.exists(self.results_csv_path):
            raise FileNotFoundError(f"File not found: {self.results_csv_path}")
        os.makedirs(self.save_dir, exist_ok=True)

        data = pd.read_csv(self.results_csv_path)
        
        # remove any rows with NaN values
        data = data.dropna()
        y_target = data[self.target_variable]
        X_parameters = data[self.parameters]
        self.model = PySRRegressor(
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            parsimony=0.0032,
            turbo=True,
            niterations=self.n_iterations,
            population_size=self.population_size,
            # extra_sympy_mappings=self.additional_operators,
            output_directory=self.save_dir,
            run_id=self._get_id()
            )
        self.model.fit(X=X_parameters, y=y_target)
        return self.model
    
def derive(
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
    model = Derive(
        evolve_dir_path=evolve_dir_path,
        target_variable=target_variable,
        parameters=parameters,
        save_dir=save_dir,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        n_iterations=n_iterations,
        population_size=population_size,
        #additional_operators=additional_operators
    )
    return model.run_pysr()