import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import griddata
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

class Plotting:
    @staticmethod
    def plot_epochs(evolve_dir_path, show=True, save_dir=None, save_ext=".png"):

        epochs_csv_path = os.path.join(evolve_dir_path, "epochs.csv")
        results_csv_path = os.path.join(evolve_dir_path, "results.csv")
        save_dir = save_dir if save_dir else os.path.join(evolve_dir_path, "figures")
        save_ext = save_ext.strip(".") if save_ext else "png"
        os.makedirs(save_dir, exist_ok=True)

        if save_ext not in ["png", "jpg", "jpeg", "pdf", "svg"]:
            raise ValueError("Invalid save_ext. Must be one of 'png', 'jpg', 'jpeg', 'pdf', or 'svg'.")
        if not os.path.exists(epochs_csv_path):
            raise FileNotFoundError(f"File not found: {epochs_csv_path}")
        if not os.path.exists(results_csv_path):
            raise FileNotFoundError(f"File not found: {results_csv_path}")

        epochs_data = pd.read_csv(epochs_csv_path)
        results_data = pd.read_csv(results_csv_path)
        mean_cols = [col for col in epochs_data.columns if col.lower().startswith("mean")]
        sigma_cols = [col for col in epochs_data.columns if col.lower().startswith("sigma")]
        epoch_col = [col for col in epochs_data.columns if col.lower().startswith("epoch")][0]
        results_cols = [col for col in results_data.columns if any(col.lower() in epoch_col.lower().split("mean ") for epoch_col in mean_cols)]
        results_epoch_col = [col for col in results_data.columns if col.lower().startswith("epoch")][0]

        for mean_col, sigma_col, results_col in zip(mean_cols, sigma_cols, results_cols):
            plt.plot(
                epochs_data[epoch_col],
                epochs_data[mean_col],
                label=mean_col,
                color="black"
            )
            plt.scatter(
                results_data[results_epoch_col],
                results_data[results_col],
                label=results_col,
                marker="o",
                alpha=0.5,
                c="black",
                s=8,
                edgecolors="black",
                facecolor="none"
            )
            plt.fill_between(
                epochs_data[epoch_col],
                epochs_data[mean_col] - epochs_data[sigma_col],
                epochs_data[mean_col] + epochs_data[sigma_col],
                alpha=0.2,
                color = "black"
            )
            plt.xlabel(epoch_col)
            plt.ylabel(mean_col)
            plt.title(f"{mean_col} vs {epoch_col}")
            plt.legend()
            
            file_name = f"{mean_col}_vs_{epoch_col}.{save_ext}"
            plt.savefig(os.path.join(save_dir, file_name))
            if show:
                plt.show()
            plt.close()
    
    @staticmethod
    def plot_vars(evolve_dir_path, x:str, y:str, z: str=None, cval: str=None, show=True, save_dir=None, save_ext=".png", cmap="viridis"):
        """
        Plots the given variables against each other.
        if only x and y are provided, plots x vs y.
        if x, y, and cval are provided but not z, plots 2-d histogram x vs y vs cval.
        if x, y, and z are provided but not cval, plots 3-d surface of x vs y vs z.
        if x, y, z, and cval are provided, plots 3-d surface of x vs y vs z with cval as color.
        """

        results_csv_path = os.path.join(evolve_dir_path, "results.csv")
        save_dir = save_dir if save_dir else os.path.join(evolve_dir_path, "figures")
        save_ext = save_ext.strip(".") if save_ext else "png"
        os.makedirs(save_dir, exist_ok=True)

        if save_ext not in ["png", "jpg", "jpeg", "pdf", "svg"]:
            raise ValueError("Invalid save_ext. Must be one of 'png', 'jpg', 'jpeg', 'pdf', or 'svg'.")
        if not os.path.exists(results_csv_path):
            raise FileNotFoundError(f"File not found: {results_csv_path}")
        
        # Read the csv file
        data = pd.read_csv(results_csv_path)

        if z is None and cval is None:
            plt.scatter(
                data[x],
                data[y],
                marker="o",
                c="black",
                s=8,
                alpha=0.5
            )
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{x} vs {y}")
            file_name = f"{x}_vs_{y}.{save_ext}"
            plt.savefig(os.path.join(save_dir, file_name))
            if show:
                plt.show()
            plt.close()
            
        elif cval is not None and z is None:
            # Voronoi plot with cval as color
            fig, ax = plt.subplots()
            ax = Plotting._plot_voronoi(data, x, y, cval, cmap=cmap, ax=ax) 
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"Voronoi Plot of {x} vs {y} colored by {cval}")
            file_name = f"{x}_vs_{y}_voronoi_{cval}.{save_ext}"
            plt.savefig(os.path.join(save_dir, file_name))
            if show:
                plt.show()
            plt.close()

        elif z is not None and cval is None:
            # 3-D surface plot using Plotly
            x_values = data[x].values
            y_values = data[y].values
            z_values = data[z].values

            xi, yi = np.meshgrid(np.linspace(x_values.min(), x_values.max(), 50),
                                 np.linspace(y_values.min(), y_values.max(), 50))
            zi = griddata((x_values, y_values), z_values, (xi, yi), method='linear')
            fig = go.Figure(data=[go.Surface(
                z=zi,
                x=xi,
                y=yi,
                colorscale=cmap,
                colorbar=dict(title=z),
                hoverinfo='all'
                )])

            fig.update_layout(
                title=f"{x} vs {y} vs {z}",
                scene=dict(
                    xaxis_title=x,
                    yaxis_title=y,
                    zaxis_title=z,
                ),
                margin=dict(l=20, r=20, b=20, t=50)  # Adjust margins
            )
            # Save and show the plot
            file_name = f"{x}_vs_{y}_vs_{z}.html"
            fig.write_html(os.path.join(save_dir, file_name))  # Save as HTML

            if show:
                fig.show()


        elif z is not None and cval is not None:
            # 3-D surface plot with color
            x_values = data[x].values
            y_values = data[y].values
            z_values = data[z].values
            c_values = data[cval].values  # Color values

            xi, yi = np.meshgrid(np.linspace(x_values.min(), x_values.max(), 50),
                                 np.linspace(y_values.min(), y_values.max(), 50))
            zi = griddata((x_values, y_values), z_values, (xi, yi), method='linear')
            ci = griddata((x_values, y_values), c_values, (xi, yi), method='linear')
            
            fig = go.Figure(data=[go.Surface(
                z=zi,
                x=xi,
                y=yi,
                surfacecolor=ci,
                colorscale=cmap,
                colorbar=dict(title=cval),  # Add colorbar title
                )])
            fig.update_layout(
                title=f"{x} vs {y} vs {z} vs {cval})",
                scene=dict(
                    xaxis_title=x,
                    yaxis_title=y,
                    zaxis_title=z,
                ),
                margin=dict(l=20, r=20, b=20, t=50)  # Adjust margins
            )

            # Save and show the plot
            file_name = f"{x}_vs_{y}_vs_{z}_color_{cval}.html"
            fig.write_html(os.path.join(save_dir, file_name))  # Save as HTML
            
            if show:
                fig.show()
        else:
            raise ValueError("Invalid input. x and y must be provided. z and cval are optional.")
    
    @staticmethod
    def _plot_voronoi(data, x, y, cval, cmap="viridis", ax=None):
        """
        Plots a Voronoi diagram with regions colored by a given value.

        Args:
            data (pd.DataFrame): DataFrame containing the data.
            x (str): Column name for x-coordinates.
            y (str): Column name for y-coordinates.
            cval (str): Column name for the value to color the regions by.
            cmap (str, optional): Colormap to use. Defaults to "viridis".
            ax (matplotlib.axes._axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created. Defaults to None.

        Returns:
            matplotlib.axes._axes.Axes: The Axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Calculate Bounds
        x_min, x_max = data[x].min(), data[x].max()
        y_min, y_max = data[y].min(), data[y].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_margin = x_range * 0.2
        y_margin = y_range * 0.2

        x_lower = x_min - x_margin
        x_upper = x_max + x_margin
        y_lower = y_min - y_margin
        y_upper = y_max + y_margin

        # Create Boundary Points
        boundary_points = [
            [x_lower, y_lower],
            [x_lower, y_upper],
            [x_upper, y_lower],
            [x_upper, y_upper],
            [x_lower, (y_lower + y_upper) / 2],
            [x_upper, (y_lower + y_upper) / 2],
            [(x_lower + x_upper) / 2, y_lower],
            [(x_lower + x_upper) / 2, y_upper],
        ]

        # Assign cval Values
        boundary_cval = data[cval].mean()
        boundary_cvals = [boundary_cval] * len(boundary_points)

        # Append to Data
        boundary_df = pd.DataFrame(boundary_points, columns=[x, y])
        boundary_df[cval] = boundary_cvals
        data = pd.concat([data, boundary_df], ignore_index=True)

        points = data[[x, y]].values  # Extract x and y coordinates
        vor = Voronoi(points)

        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            line_colors='black',
            line_width=0.5,
            line_alpha=0.5,
            show_points=False,
            point_size=2,
            )

        # Color the Voronoi regions based on cval
        min_cval = data[cval].min()
        max_cval = data[cval].max()

        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                norm_cval = (data[cval].iloc[r] - min_cval) / (max_cval - min_cval)
                ax.fill(*zip(*polygon), color=plt.cm.get_cmap(cmap)(norm_cval), alpha=0.85)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data[cval].min(), vmax=data[cval].max()))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(cval)
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        return ax