import pandas as pd
import matplotlib.pyplot as plt

class Plotting:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None

    def read_csv(self):
        self.data = pd.read_csv(self.csv_path)

    def plot_epochs(self):
        mean_cols = [col for col in self.data.columns if col.lower().startswith("mean")]
        sigma_cols = [col for col in self.data.columns if col.lower().startswith("sigma")]
        epoch_col = "Epoch"
        
        print(mean_cols)
        print(sigma_cols)
        for mean_col, sigma_col, in zip(mean_cols, sigma_cols):
            plt.plot(self.data[epoch_col], self.data[mean_col], label=mean_col)
            plt.fill_between(
                self.data[epoch_col],
                self.data[mean_col] - self.data[sigma_col],
                self.data[mean_col] + self.data[sigma_col],
                alpha=0.2,
            )
            plt.xlabel(epoch_col)
            plt.ylabel(mean_col)
            plt.title(f"{mean_col} vs {epoch_col}")
            plt.legend()
            plt.show()

    def go(self):
        self.read_csv()
        self.plot_epochs()


# Example usage:
if __name__ == "__main__":
    csv_path = r"C:\Users\Rob.Hart-Villamil\Documents\python_project_files\gmsh_project\Genesis\evolve_1\epochs.csv"
    plotting = Plotting(csv_path)
    plotting.go()