"""Visualization utilities for evolutionary optimization results.

This module provides visualization tools for analyzing optimization results from the evopt package.
It contains functionality for plotting parameter evolution across epochs, convergence metrics,
and exploring parameter spaces through various visualization techniques including 2D scatter plots,
Voronoi diagrams, and 3D surface plots.

Examples:
    Plot the evolution of parameters across epochs:
    
    >>> from evopt.plotting import Plotting
    >>> Plotting.plot_epochs("path/to/evolve_dir")
    
    Visualize relationships between two parameters:
    
    >>> Plotting.plot_vars("path/to/evolve_dir", "param1", "param2")
"""

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import griddata
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
import os

class Plotting:
    """Visualization tools for evolutionary optimization results.
        
    This class provides static methods for visualizing results from evolutionary optimization
    runs. It can generate plots showing the evolution of parameters across epochs, parameter
    convergence, and explore relationships between parameters through various visualization
    techniques.
    
    The class operates on the CSV result files produced during optimization runs, which contain
    information about individual evaluations and epoch statistics.
    
    Examples:
        Create epoch plots showing parameter evolution:
        
        >>> Plotting.plot_epochs("path/to/evolve_dir")
        
        Create a 1-D scatterplot visualization of two parameters:
        
        >>> Plotting.plot_vars("path/to/evolve_dir", x = "param1", y = "param2")
        
        Create a 2-D Voronoi plot visualization with parameter values:
        
        >>> Plotting.plot_vars("path/to/evolve_dir", x = "param1", y = "param2", cval = "error")

        Create a 3-D surface plot visualization with parameter values:

        >>> Plotting.plot_vars("path/to/evolve_dir", x = "param1", y = "param2", z = "param3")

        Create a 3-D surface plot visualization with parameter values and color:

        >>> Plotting.plot_vars("path/to/evolve_dir", x = "param1", y = "param2", z = "param3", cval = "error")
    """

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
        """Plot the mean and sigma values for each parameter across epochs.
        
        Creates visualizations showing how parameters evolved during optimization. For each parameter,
        this method generates a plot showing:
        - The mean value across epochs (line)
        - The individual evaluation results (scattered points)
        - The standard deviation ranges (shaded areas)
        
        Additionally, generates a convergence plot showing normalized sigma values.
        
        Args:
            evolve_dir_path: Path to the directory containing the evolution data files.
            show: Whether to display the plots in the current interface.
            save_dir: Directory to save the plot files. If None, creates a 'figures' 
                subdirectory in evolve_dir_path.
            save_ext: File extension for saved plots. Must be one of 'png', 'jpg', 
                'jpeg', 'pdf', or 'svg'.
            cmap: Matplotlib colormap name to use for the plots.
            point_alpha: Opacity of the scattered points (0.0-1.0).
            shade_alpha: Opacity of the standard deviation shaded areas (0.0-1.0).
            save_figures: Whether to save the generated figures to disk.
            
        Returns:
            None
            
        Raises:
            FileNotFoundError: If the required CSV files are not found.
            ValueError: If an invalid file extension is provided.
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

        results_cols = []
        for mean_col in mean_cols:
            base_name = mean_col.split("mean ")[-1].strip()
            for col in results_data.columns:
                if base_name.lower() in col.lower():
                    results_cols.append(col)
                    break
        #results_cols = [col for col in results_data.columns if any(col.lower() in epoch_col.lower().split("mean ") for epoch_col in mean_cols)]
        results_epoch_col = [col for col in results_data.columns if col.lower().startswith("epoch")][0]

        # assign a colour to each column
        colours = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(mean_cols)))
        for mean_col, sigma_col, results_col, colour in zip(mean_cols, sigma_cols, results_cols, colours):
            plt.figure(figsize=(6, 4))
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
        plt.figure(figsize=(6, 4))
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
        save_figures: bool = True,
        title: str = None
    ):
        """Visualize relationships between optimization parameters and results.
        
        Creates visualizations to explore the relationships between parameters and results.
        The visualization type depends on the provided parameters:
        
        - If only x and y are provided: Creates a 2D scatter plot
        - If x, y, and cval are provided: Creates a Voronoi diagram with regions colored by cval
        - If x, y, and z are provided: Creates a 3D surface plot of x, y, z
        - If x, y, z, and cval are provided: Creates a 3D surface with color determined by cval
        
        Args:
            evolve_dir_path: Path to the directory containing evolution data.
            x: Column name for x-axis values.
            y: Column name for y-axis values.
            z: Column name for z-axis values (for 3D plots). Defaults to None.
            cval: Column name for values used to color the plot. Defaults to None.
            show: Whether to display the plots in the current interface.
            save_dir: Directory to save the plots. If None, creates a 'figures' 
                subdirectory in evolve_dir_path.
            save_ext: File extension for saved plots ('png', 'jpg', 'jpeg', 'pdf', 'svg').
                For 3D plots, always saved as HTML regardless of this setting.
            cmap: Colormap name to use for the plots.
            point_colour: Color for scatter points.
            alpha: Opacity of plot elements (0.0-1.0).
            save_figures: Whether to save the generated figures to disk.
            
        Returns:
            Either a matplotlib Axes object (for 2D plots) or a plotly Figure object (for 3D plots).
            
        Raises:
            FileNotFoundError: If the required CSV files are not found.
            ValueError: If invalid parameters are provided or file extension is invalid.
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

        def safe_eval(expression, data):
            try:
                return data.eval(expression)
            except (ValueError, SyntaxError, NameError, TypeError) as e:
                print(f"Error evaluating expression '{expression}': {e}")
                return None

        x_values = safe_eval(x, data)
        y_values = safe_eval(y, data)
        z_values = safe_eval(z, data) if z else None
        c_values = safe_eval(cval, data) if cval else None

        if x_values is None or y_values is None:
            raise ValueError("Invalid x or y expression.")
        
        # Sanitize x, y, cval for filename
        x_sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', x)
        y_sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', y)
        z_sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', z) if z else None
        cval_sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', cval) if cval else None

        if z is None and cval is None:
            title = title if title else f"{x} vs {y}"
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                x_values,
                y_values,
                marker="o",
                c=point_colour,
                s=8,
                alpha=alpha
            )
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(title)
            file_name = f"{x_sanitized}_vs_{y_sanitized}.{save_ext}"
            if save_figures:
                plt.savefig(os.path.join(save_dir, file_name))
            
            if show:
                plt.show()
            plt.close()
            return ax

            
        elif cval is not None and z is None:
            title = title if title else f"Voronoi Plot of {x} vs {y} colored by {cval}"
            # Voronoi plot with cval as color
            fig, ax = plt.subplots(figsize=(6, 4))
            # Create a temporary DataFrame for the Voronoi plot
            temp_data = pd.DataFrame({'x': x_values, 'y': y_values, 'cval': c_values})
            
            ax = Plotting._plot_voronoi(
                temp_data, 'x', 'y', 'cval',
                cmap=cmap,
                ax=ax,
                clip_infinite=True,
                point_colour=point_colour,
                alpha=alpha
            ) 
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(title)
            file_name = f"{x_sanitized}_vs_{y_sanitized}_vs_{cval_sanitized}_Voronoi.{save_ext}"
            if save_figures:
                plt.savefig(os.path.join(save_dir, file_name))
            
            if show:
                plt.show()
            plt.close()
            return ax

        elif z is not None and cval is None:
            title = title if title else f"{x} vs {y} vs {z}"
            # 3-D surface plot using Plotly

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
                title=title,
                scene=dict(
                    xaxis_title=x,
                    yaxis_title=y,
                    zaxis_title=z,
                    aspectratio=dict(x=1, y=1, z=1),  # Adjust aspect ratio
                ),
                margin=dict(l=20, r=20, b=20, t=50)  # Adjust margins
            )
            # Save and show the plot
            file_name = f"{x_sanitized}_vs_{y_sanitized}_vs_{z_sanitized}_surface.html"
            if save_figures:
                fig.write_html(os.path.join(save_dir, file_name))  # Save as HTML

            if show:
                fig.show()
            return fig


        elif z is not None and cval is not None:
            # 3-D surface plot with color
            title = title if title else f"{x} vs {y} vs {z} vs {cval}"

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
                title=title,
                scene=dict(
                    xaxis_title=x,
                    yaxis_title=y,
                    zaxis_title=z,
                    aspectratio=dict(x=1, y=1, z=1),
                ),
                margin=dict(l=20, r=20, b=20, t=50)  # Adjust margins
            )

            # Save and show the plot
            file_name = f"{x_sanitized}_vs_{y_sanitized}_vs_{z_sanitized}_vs_{cval_sanitized}_surface.html"
            if save_figures:
                fig.write_html(os.path.join(save_dir, file_name))  # Save as HTML
            if show:
                fig.show()
            return fig
        else:
            raise ValueError("Invalid input. x and y must be provided. z and cval are optional.")
    
    @staticmethod
    def _plot_voronoi(data, x, y, cval, cmap="viridis", ax=None, clip_infinite=True, point_colour="black", alpha=0.25):
        """Create a Voronoi diagram with regions colored by a specified value.
        
        This helper method generates a Voronoi tessellation of the parameter space,
        where each region is colored according to a specified value (e.g., error).
        
        Args:
            data: DataFrame containing the data points.
            x: Column name for x-coordinates.
            y: Column name for y-coordinates.
            cval: Column name for the value to color the regions by.
            cmap: Colormap name to use for coloring regions.
            ax: Matplotlib axes object to plot on. If None, creates a new figure.
            clip_infinite: Whether to clip infinite Voronoi regions by adding boundary points.
            point_colour: Color for the data points.
            alpha: Opacity for the data points (0.0-1.0).
            
        Returns:
            matplotlib.axes._axes.Axes: The axes object with the Voronoi plot.
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