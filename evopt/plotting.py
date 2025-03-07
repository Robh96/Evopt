import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import griddata
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

class Plotting:
    @staticmethod
    def plot_epochs(
        evolve_dir_path: str,
        show: bool = True,
        save_dir: str = None,
        save_ext: str = ".png",
        cmap: str = "Dark2",
        point_alpha: float|int = 0.75,
        shade_alpha: float|int = 0.4,
        save_figures: bool = True
    ):
        """
        Plots the mean and sigma values for each parameter across epochs.
        Also plots the results for each epoch.
        Args:
            evolve_dir_path (str): Path to the directory containing the evolution data.
            show (bool, optional): Whether to display the plots. Defaults to True.
            save_dir (str, optional): Directory to save the plots. Defaults to None.
            save_ext (str, optional): File extension for saving the plots. Defaults to ".png".
            cmap (str, optional): Colormap to use. Defaults to "Dark2".
            point_alpha (float, optional): Alpha value for the points. Defaults to 0.75.
            shade_alpha (float, optional): Alpha value for the shaded regions. Defaults to 0.4.
            save_figures (bool, optional): Whether to save the figures. Defaults to True.
        """

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
        norm_sigma_cols = [col for col in epochs_data.columns if col.lower().startswith("norm sigma")]
        epoch_col = [col for col in epochs_data.columns if col.lower().startswith("epoch")][0]
        results_cols = [col for col in results_data.columns if any(col.lower() in epoch_col.lower().split("mean ") for epoch_col in mean_cols)]
        results_epoch_col = [col for col in results_data.columns if col.lower().startswith("epoch")][0]

        # assign a colour to each column
        colours = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(mean_cols)))
        for mean_col, sigma_col, results_col, colour in zip(mean_cols, sigma_cols, results_cols, colours):
            plt.plot(
                epochs_data[epoch_col],
                epochs_data[mean_col],
                label=mean_col,
                color=colour
            )
            plt.scatter(
                results_data[results_epoch_col],
                results_data[results_col],
                label=results_col,
                marker="o",
                alpha=point_alpha,
                color=colour,
                s=8,
                edgecolors=colour,
                facecolor="none"
            )
            plt.fill_between(
                epochs_data[epoch_col],
                epochs_data[mean_col] - epochs_data[sigma_col],
                epochs_data[mean_col] + epochs_data[sigma_col],
                alpha=shade_alpha,
                color = colour
            )
            plt.xlabel(epoch_col)
            plt.ylabel(mean_col)
            plt.title(f"{mean_col} vs {epoch_col}")
            plt.legend()
            
            file_name = f"{mean_col}_vs_{epoch_col}.{save_ext}"
            if save_figures:
                plt.savefig(os.path.join(save_dir, file_name))
            if show:
                plt.show()
            plt.close()
    
        # Plot each normalized sigma column with its corresponding color
        for i, norm_sigma_col in enumerate(norm_sigma_cols):
            color = colours[i]  # Get the color for this line
            plt.plot(
                epochs_data[epoch_col],
                epochs_data[norm_sigma_col],
                label=norm_sigma_col,
                color=color,
            )
        plt.xlabel(epoch_col)
        plt.ylabel("Normalised Sigma")
        plt.title(f"Convergence Plot")
        plt.legend()

        file_name = f"convergence_plot.{save_ext}"
        if save_figures:
            plt.savefig(os.path.join(save_dir, file_name))
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_vars(
        evolve_dir_path: str,
        x: str,
        y: str,
        z: str = None,
        cval: str = None,
        show: bool = True,
        save_dir: str = None,
        save_ext: str = ".png",
        cmap: str = "viridis",
        point_colour: str = "black",
        alpha: float|int = 1,
        save_figures: bool = True
    ):
        """
        Plots the given variables against each other.
        if only x and y are provided, plots x vs y.
        if x, y, and cval are provided but not z, plots 2-d histogram x vs y vs cval.
        if x, y, and z are provided but not cval, plots 3-d surface of x vs y vs z.
        if x, y, z, and cval are provided, plots 3-d surface of x vs y vs z with cval as color.

        Args:
            evolve_dir_path (str): Path to the directory containing the evolution data.
            x (str): Column name for x-coordinates.
            y (str): Column name for y-coordinates.
            z (str, optional): Column name for z-coordinates. Defaults to None.
            cval (str, optional): Column name for the value to color the regions by. Defaults to None.
            show (bool, optional): Whether to display the plots. Defaults to True.
            save_dir (str, optional): Directory to save the plots. Defaults to None.
            save_ext (str, optional): File extension for saving the plots. Defaults to ".png".
            cmap (str, optional): Colormap to use. Defaults to "viridis".
            point_colour (str, optional): Colour of the points. Defaults to "black".
            alpha (float, optional): Alpha value for the points. Defaults to 1.
            save_figures (bool, optional): Whether to save the figures. Defaults to True.
        """

        results_csv_path = os.path.join(evolve_dir_path, "results.csv")
        save_dir = save_dir if save_dir else os.path.join(evolve_dir_path, "figures")
        save_ext = save_ext.strip(".") if save_ext else "png"
        if save_figures:
            os.makedirs(save_dir, exist_ok=True)

        if save_ext not in ["png", "jpg", "jpeg", "pdf", "svg"]:
            raise ValueError("Invalid save_ext. Must be one of 'png', 'jpg', 'jpeg', 'pdf', or 'svg'.")
        if not os.path.exists(results_csv_path):
            raise FileNotFoundError(f"File not found: {results_csv_path}")
        
        # Read the csv file
        data = pd.read_csv(results_csv_path)

        if z is None and cval is None:
            fig, ax = plt.subplots()
            ax.scatter(
                data[x],
                data[y],
                marker="o",
                c=point_colour,
                s=8,
                alpha=alpha
            )
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{x} vs {y}")
            file_name = f"{x}_vs_{y}.{save_ext}"
            if save_figures:
                plt.savefig(os.path.join(save_dir, file_name))
            if show:
                plt.show()
            plt.close()
            return ax

            
        elif cval is not None and z is None:
            # Voronoi plot with cval as color
            fig, ax = plt.subplots()
            ax = Plotting._plot_voronoi(
                data, x, y, cval,
                cmap=cmap,
                ax=ax,
                clip_infinite=True,
                point_colour=point_colour,
                alpha=alpha
            ) 
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"Voronoi Plot of {x} vs {y} colored by {cval}")
            file_name = f"{x}_vs_{y}_vs_{cval}_Voronoi.{save_ext}"
            if save_figures:
                plt.savefig(os.path.join(save_dir, file_name))
            if show:
                plt.show()
            plt.close()
            return ax

        elif z is not None and cval is None:
            # 3-D surface plot using Plotly
            x_values = data[x].values
            y_values = data[y].values
            z_values = data[z].values

            xi, yi = np.meshgrid(np.linspace(x_values.min(), x_values.max(), 100),
                                 np.linspace(y_values.min(), y_values.max(), 100))
            zi = griddata((x_values, y_values), z_values, (xi, yi), method='linear')
            fig = go.Figure(data=[go.Surface(
                z=zi,
                x=xi,
                y=yi,
                opacity=alpha,
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
                    aspectratio=dict(x=1, y=1, z=1),  # Adjust aspect ratio
                ),
                margin=dict(l=20, r=20, b=20, t=50)  # Adjust margins
            )
            # Save and show the plot
            file_name = f"{x}_vs_{y}_vs_{z}_surface.html"
            if save_figures:
                fig.write_html(os.path.join(save_dir, file_name))  # Save as HTML

            if show:
                fig.show()
            return fig


        elif z is not None and cval is not None:
            # 3-D surface plot with color
            x_values = data[x].values
            y_values = data[y].values
            z_values = data[z].values
            c_values = data[cval].values  # Color values

            xi, yi = np.meshgrid(np.linspace(x_values.min(), x_values.max(), 100),
                                 np.linspace(y_values.min(), y_values.max(), 100))
            zi = griddata((x_values, y_values), z_values, (xi, yi), method='linear')
            ci = griddata((x_values, y_values), c_values, (xi, yi), method='linear')
            
            fig = go.Figure(data=[go.Surface(
                z=zi,
                x=xi,
                y=yi,
                surfacecolor=ci,
                opacity=alpha,
                colorscale=cmap,
                colorbar=dict(title=cval),  # Add colorbar title
                )])
            fig.update_layout(
                title=f"{x} vs {y} vs {z} vs {cval}",
                scene=dict(
                    xaxis_title=x,
                    yaxis_title=y,
                    zaxis_title=z,
                    aspectratio=dict(x=1, y=1, z=1),
                ),
                margin=dict(l=20, r=20, b=20, t=50)  # Adjust margins
            )

            # Save and show the plot
            file_name = f"{x}_vs_{y}_vs_{z}_vs_{cval}_surface.html"
            if save_figures:
                fig.write_html(os.path.join(save_dir, file_name))  # Save as HTML
            if show:
                fig.show()
            return fig
        else:
            raise ValueError("Invalid input. x and y must be provided. z and cval are optional.")
    
    @staticmethod
    def _plot_voronoi(data, x, y, cval, cmap="viridis", ax=None, clip_infinite=True, point_colour="black", alpha=0.25):
        """
        Plots a Voronoi diagram with regions colored by a given value.

        Args:
            data (pd.DataFrame): DataFrame containing the data.
            x (str): Column name for x-coordinates.
            y (str): Column name for y-coordinates.
            cval (str): Column name for the value to color the regions by.
            cmap (str, optional): Colormap to use. Defaults to "viridis".
            ax (matplotlib.axes._axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created. Defaults to None.
            clip_infinite (bool, optional): Whether to clip the infinite regions. Defaults to True.
        Returns:
            matplotlib.axes._axes.Axes: The Axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Calculate Bounds
        x_min, x_max = data[x].min(), data[x].max()
        y_min, y_max = data[y].min(), data[y].max()

        if clip_infinite:
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_margin = x_range * 0.5  # Adjust for desired margin
            y_margin = y_range * 0.5  

            x_lower = x_min - x_margin
            x_upper = x_max + x_margin
            y_lower = y_min - y_margin
            y_upper = y_max + y_margin

            # Create More Boundary Points
            num_boundary_points = 20  # Adjust for desired density
            x_coords_lower = np.linspace(x_lower, x_upper, num_boundary_points)
            x_coords_upper = np.linspace(x_lower, x_upper, num_boundary_points)
            y_coords_lower = np.linspace(y_lower, y_upper, num_boundary_points)
            y_coords_upper = np.linspace(y_lower, y_upper, num_boundary_points)

            boundary_points = []
            for x_coord in x_coords_lower:
                boundary_points.append([x_coord, y_lower])  # Bottom edge
            for x_coord in x_coords_upper:
                boundary_points.append([x_coord, y_upper])  # Top edge
            for y_coord in y_coords_lower:
                boundary_points.append([x_lower, y_coord])  # Left edge
            for y_coord in y_coords_upper:
                boundary_points.append([x_upper, y_coord])  # Right edge

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
        )
        
        # Color the Voronoi regions based on cval
        min_cval = data[cval].min()
        max_cval = data[cval].max()

        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                norm_cval = (data[cval].iloc[r] - min_cval) / (max_cval - min_cval)
                ax.fill(*zip(*polygon), color=plt.cm.get_cmap(cmap)(norm_cval), alpha=1)
        
        ax.scatter(
            vor.points[:, 0],
            vor.points[:, 1],
            c=point_colour,
            alpha=alpha,
            s=7
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data[cval].min(), vmax=data[cval].max()))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(cval)
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        return ax