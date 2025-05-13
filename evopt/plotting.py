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
import matplotlib.ticker as mticker
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

        # Drop rows containing NaN values from x_values, y_values, z_values, c_values
        valid_mask = pd.notnull(x_values) & pd.notnull(y_values)
        if z_values is not None:
            valid_mask = valid_mask & pd.notnull(z_values)
        if c_values is not None:
            valid_mask = valid_mask & pd.notnull(c_values)
            
        # Apply the mask to all values
        x_values = x_values[valid_mask]
        y_values = y_values[valid_mask]
        if z_values is not None:
            z_values = z_values[valid_mask]
        if c_values is not None:
            c_values = c_values[valid_mask]

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

            

            # original_points_norm_df = pd.DataFrame({
            #     'x_norm': x_values_norm, 
            #     'y_norm': y_values_norm, 
            #     'cval': c_values
            # })
            
            # ax = Plotting._plot_voronoi(
            #     original_points_df=original_points_norm_df, 
            #     x_col_norm='x_norm', y_col_norm='y_norm', cval_col='cval',
            #     x_orig_min=x_orig_min, x_orig_max=x_orig_max,
            #     y_orig_min=y_orig_min, y_orig_max=y_orig_max,
            #     cmap=cmap, ax=ax, clip_infinite=True,
            #     point_colour=point_colour, alpha=alpha
            # )

            # Create a temporary DataFrame for the Voronoi plot
            temp_data = pd.DataFrame({x: x_values, y: y_values, cval: c_values})
            
            ax = Plotting._plot_voronoi(
                temp_data, x, y, cval,
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
    def _plot_voronoi(data, x, y, cval, 
                      cmap="viridis", ax=None, clip_infinite=True, point_colour="black", alpha=0.25):
        """Create a Voronoi diagram with regions colored by a specified value.
        Normalization is handled internally. Axis ticks show original data scales.
        """
        
        if ax is None:
            fig, ax = plt.subplots()

        # 1. Store original min/max for tick relabeling
        x_orig_min, x_orig_max = data[x].min(), data[x].max()
        y_orig_min, y_orig_max = data[y].min(), data[y].max()

        x_range_orig = x_orig_max - x_orig_min
        y_range_orig = y_orig_max - y_orig_min

        # 2. Normalize original x and y values
        if abs(x_range_orig) < 1e-9: # Check for zero range
            x_norm = pd.Series(0.5, index=data.index)
        else:
            x_norm = (data[x] - x_orig_min) / x_range_orig
        
        if abs(y_range_orig) < 1e-9: # Check for zero range
            y_norm = pd.Series(0.5, index=data.index)
        else:
            y_norm = (data[y] - y_orig_min) / y_range_orig

        # 3. Create a DataFrame with normalized coordinates for the original points
        # This df's 'x_norm_col', 'y_norm_col' will be used for scattering original points
        original_points_norm_df = pd.DataFrame({
            'x_norm_col': x_norm, # Using a distinct name for the column in this temp df
            'y_norm_col': y_norm, # Using a distinct name for the column in this temp df
            cval: data[cval] # Keep original cval name and values
        })

        # This DataFrame will be used as input to Voronoi, potentially augmented with boundary points
        data_for_voronoi_calc = original_points_norm_df.copy()

        # Normalized bounds from original data points (typically 0 to 1)
        x_norm_min_core, x_norm_max_core = original_points_norm_df['x_norm_col'].min(), original_points_norm_df['x_norm_col'].max()
        y_norm_min_core, y_norm_max_core = original_points_norm_df['y_norm_col'].min(), original_points_norm_df['y_norm_col'].max()
        
        # Default plot limits in normalized coordinates
        display_xlim_norm = [x_norm_min_core, x_norm_max_core]
        display_ylim_norm = [y_norm_min_core, y_norm_max_core]
        boundary_gen_xlim_norm = list(display_xlim_norm) 
        boundary_gen_ylim_norm = list(display_ylim_norm)

        if clip_infinite:
            x_norm_range_core = x_norm_max_core - x_norm_min_core
            y_norm_range_core = y_norm_max_core - y_norm_min_core
            
            x_norm_range_core = x_norm_range_core if x_norm_range_core > 1e-9 else 1.0
            y_norm_range_core = y_norm_range_core if y_norm_range_core > 1e-9 else 1.0

            x_margin_norm = x_norm_range_core * 0.5
            y_margin_norm = y_norm_range_core * 0.5  

            # Expand limits for boundary generation
            x_lower_bound_norm_for_boundary = x_norm_min_core - x_margin_norm
            x_upper_bound_norm_for_boundary = x_norm_max_core + x_margin_norm
            y_lower_bound_norm_for_boundary = y_norm_min_core - y_margin_norm
            y_upper_bound_norm_for_boundary = y_norm_max_core + y_margin_norm


            boundary_gen_xlim_norm = [x_lower_bound_norm_for_boundary, x_upper_bound_norm_for_boundary]
            boundary_gen_ylim_norm = [y_lower_bound_norm_for_boundary, y_upper_bound_norm_for_boundary]

            num_boundary_pts_per_edge = 20
            # Use the expanded limits for linspace
            b_pts_x_bottom_top = np.linspace(boundary_gen_xlim_norm[0], boundary_gen_xlim_norm[1], num_boundary_pts_per_edge)
            b_pts_y_left_right = np.linspace(boundary_gen_ylim_norm[0], boundary_gen_ylim_norm[1], num_boundary_pts_per_edge)

            boundary_points_list_norm = []
            for x_coord_b in b_pts_x_bottom_top: boundary_points_list_norm.append([x_coord_b, boundary_gen_ylim_norm[0]]) # y_lower_bound_norm
            for x_coord_b in b_pts_x_bottom_top: boundary_points_list_norm.append([x_coord_b, boundary_gen_ylim_norm[1]]) # y_upper_bound_norm
            for y_coord_b in b_pts_y_left_right:  boundary_points_list_norm.append([boundary_gen_xlim_norm[0], y_coord_b]) # x_lower_bound_norm
            for y_coord_b in b_pts_y_left_right: boundary_points_list_norm.append([boundary_gen_xlim_norm[1], y_coord_b]) # x_upper_bound_norm
            
            boundary_points_list_norm = list(set(map(tuple, boundary_points_list_norm)))
            boundary_points_list_norm = [list(pt) for pt in boundary_points_list_norm]

            if boundary_points_list_norm:
                boundary_cval_fill_value = data[cval].mean()
                boundary_cvals_for_df = [boundary_cval_fill_value] * len(boundary_points_list_norm)
                
                boundary_df_norm = pd.DataFrame(boundary_points_list_norm, columns=['x_norm_col', 'y_norm_col'])
                boundary_df_norm[cval] = boundary_cvals_for_df
                data_for_voronoi_calc = pd.concat([original_points_norm_df, boundary_df_norm], ignore_index=True)
        
        voronoi_input_coordinates = data_for_voronoi_calc[['x_norm_col', 'y_norm_col']].values
        
        unique_voronoi_input_coords = set(map(tuple, voronoi_input_coordinates))
        if len(unique_voronoi_input_coords) < 4: 
            ax.text(0.5, 0.5, "Not enough unique points for Voronoi plot.", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            def x_fmt_fallback(norm_val, pos): return f"{x_orig_min + norm_val * x_range_orig:.2g}" if abs(x_range_orig) > 1e-9 else f"{x_orig_min:.2g}"
            def y_fmt_fallback(norm_val, pos): return f"{y_orig_min + norm_val * y_range_orig:.2g}" if abs(y_range_orig) > 1e-9 else f"{y_orig_min:.2g}"
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_fmt_fallback))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt_fallback))
            ax.set_xlim(display_xlim_norm) 
            ax.set_ylim(display_ylim_norm)
            return ax

        vor = Voronoi(voronoi_input_coordinates)

        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=0.5, line_alpha=0.5, show_points=False)
        
        min_cval_for_fill = data_for_voronoi_calc[cval].min()
        max_cval_for_fill = data_for_voronoi_calc[cval].max()
        range_cval_for_fill = max_cval_for_fill - min_cval_for_fill
        if abs(range_cval_for_fill) < 1e-9: range_cval_for_fill = 1.0

        for r_idx in range(len(vor.point_region)):
            region_indices = vor.regions[vor.point_region[r_idx]]
            if not -1 in region_indices and len(region_indices) > 0:
                polygon_vertices_norm = [vor.vertices[i] for i in region_indices]
                current_point_cval = data_for_voronoi_calc[cval].iloc[r_idx]
                
                color_norm_value = (current_point_cval - min_cval_for_fill) / range_cval_for_fill
                color_norm_value = np.clip(color_norm_value, 0, 1)
                
                ax.fill(*zip(*polygon_vertices_norm), color=plt.cm.get_cmap(cmap)(color_norm_value), alpha=1)
        
        ax.scatter(
            original_points_norm_df['x_norm_col'], 
            original_points_norm_df['y_norm_col'], 
            c=point_colour, alpha=alpha, s=7, zorder=10
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data[cval].min(), vmax=data[cval].max()))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(cval)

        def x_formatter_func(norm_val, pos):
            return f"{x_orig_min + norm_val * x_range_orig:.2g}" if abs(x_range_orig) > 1e-9 else f"{x_orig_min:.2g}"
        
        def y_formatter_func(norm_val, pos):
            return f"{y_orig_min + norm_val * y_range_orig:.2g}" if abs(y_range_orig) > 1e-9 else f"{y_orig_min:.2g}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_formatter_func))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_formatter_func))

        ax.set_xlim(display_xlim_norm)
        ax.set_ylim(display_ylim_norm)
            
        return ax