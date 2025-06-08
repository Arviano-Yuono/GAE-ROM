import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from matplotlib import colormaps
import matplotlib.colors as mcolors
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from src.data import scaling
import pyvista as pv
import matplotlib.tri
from typing import Optional
import torch.nn as nn

class Plot:
    def __init__(self, 
                 data_dir: Optional[str] = None, 
                 save_dir: str = 'output/',
                 train_dataset: Optional[Data] = None,
                 val_dataset: Optional[Data] = None,
                 model: Optional[nn.Module] = None):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model

    def plot_velocity_field(self, data: Data, title: str = "Velocity Field", save = False, xlim=None, ylim=None, colormap='bwr'):
        if data.pos is None or data.x is None:
            print("Error: `data.pos` or `data.x` is None. Cannot plot velocity field.")
            return

        x_coord = data.pos[:,0].detach().cpu().numpy()
        y_coord = data.pos[:,1].detach().cpu().numpy()
        vel = data.x.detach().cpu().numpy()
        
        num_components = 1 if vel.ndim == 1 else vel.shape[1]

        if num_components == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            axes = [ax1, ax2]
            comp_titles = ['Ux Component', 'Uy Component']
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            axes = [ax1]
            comp_titles = ['Velocity Component']
        
        triang = matplotlib.tri.Triangulation(x_coord, y_coord)
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))

        for i, ax in enumerate(axes):
            vel_comp = vel if num_components == 1 else vel[:, i]

            norm = mcolors.Normalize(vmin=vel_comp.min(), vmax=vel_comp.max())
            cs = ax.tricontourf(triang, vel_comp, 100, cmap=colormap, norm=norm)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(cs, cax=cax, format=fmt)
            tick_locator = MaxNLocator(nbins=3)
            cbar.locator = tick_locator
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()
            ax.set_aspect('equal', 'box')
            ax.set_title(comp_titles[i])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
        plt.suptitle(title)
        plt.tight_layout()

        if save:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            plt.savefig(os.path.join(self.save_dir, f"{title}.png"))
        else:
            plt.show()

        plt.close(fig)
    
    def plot_comparison_fields(self, SNAP, device, dataset, params, grid="vertical", comp="_U", adjust_title=None, xlim=None, ylim=None, colormap='bwr', save=False):
        """
        Plots the velocity field solution for a given snapshot, comparing ground truth and prediction.

        Args:
            SNAP: integer value indicating the snapshot to be plotted
            pred: predicted values
            dataset: dataset containing the ground truth
            params: array of shape (num_snap,), containing the parameters associated with each snapshot
            grid: str, either "horizontal" or "vertical" for subplot arrangement
            comp: str, component suffix for the plot
            adjust_title: float, optional adjustment for title position
        """
        # Create figure with 2 subplots first
        if grid == "horizontal":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            y0 = 0.7
        elif grid == "vertical":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))
            y0 = 1.1
        else:
            raise ValueError("grid argument must be 'horizontal' or 'vertical'")
        
        if adjust_title is not None:
            y0 = adjust_title

        pred, _, _ = self.model(dataset[SNAP].to(device), params.to(device))
        pred = pred.detach().cpu().numpy()

        # Get coordinates and velocity data
        xx = dataset[SNAP].pos[:,0].detach().cpu().numpy()
        yy = dataset[SNAP].pos[:,1].detach().cpu().numpy()
        vel = dataset[SNAP].x.detach().cpu().numpy()

        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        
        # Create triangulation directly from points
        triang = matplotlib.tri.Triangulation(xx, yy)
        
        # Plot ground truth
        norm1 = mcolors.Normalize(vmin=vel.min(), vmax=vel.max())
        cs1 = ax1.tricontourf(triang, vel, 100, cmap=colormap, norm=norm1)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cbar1 = plt.colorbar(cs1, cax=cax1, format=fmt)   
        
        tick_locator = MaxNLocator(nbins=3)
        cbar1.locator = tick_locator
        cbar1.ax.yaxis.set_offset_position('left')
        cbar1.update_ticks()
        ax1.set_aspect('equal', 'box')
        ax1.set_title(f'Ground Truth')
        
        if xlim is not None:
            ax1.set_xlim(xlim)
        if ylim is not None:
            ax1.set_ylim(ylim)

        # Plot prediction
        norm2 = mcolors.Normalize(vmin=pred.min(), vmax=pred.max())
        cs2 = ax2.tricontourf(triang, pred, 100, cmap=colormap, norm=norm2)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cbar2 = plt.colorbar(cs2, cax=cax2, format=fmt)
        
        tick_locator = MaxNLocator(nbins=3)
        cbar2.locator = tick_locator
        cbar2.ax.yaxis.set_offset_position('left')
        cbar2.update_ticks()
        ax2.set_aspect('equal', 'box')
        ax2.set_title(f'Prediction Results')
        if xlim is not None:
            ax2.set_xlim(xlim)
        if ylim is not None:
            ax2.set_ylim(ylim)

        # Adjust layout
        plt.tight_layout()
        fig.suptitle('Velocity Field Comparison for $\mu$ = '+str(np.around(params.detach().cpu().numpy(), 2)), y=y0)

        if save:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            plt.savefig(os.path.join(self.save_dir, f"comparison_fields_{np.around(params.detach().cpu().numpy(), 2)}.png"))
        else:
            plt.show()

        plt.close(fig)

    def plot_velocity_field_error(self, data: Data, params, device, title: str = "Velocity Error Field", save = False, xlim=None, ylim=None, colormap='bwr'):
        if data.pos is None or data.x is None:
            print("Error: `data.pos` or `data.x` is None. Cannot plot velocity field.")
            return
        
        ground_truth = data.x
        pred, _, _ = self.model(data.to(device), params.to(device))
        pred = pred
        error = ground_truth - pred
        error_data = Data(pos=data.pos, x=error).to(device)

        self.plot_velocity_field(data=error_data, title=title, save=save, xlim=xlim, ylim=ylim, colormap=colormap)
        
        
        
        
