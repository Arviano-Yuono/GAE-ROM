import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric

class Plot:
    def __init__(self, data_dir: str = None, save_dir: str = 'output/'):
        self.data_dir = data_dir
        self.save_dir = save_dir

    def plot_loss(self, loss_list: list, title: str = "Loss"):
        plt.figure(figsize=(10, 5))

    def plot_tensor(self, tensor: torch.Tensor, data_dim: tuple, title: str = "Tensor", save = False):
        plt.figure(figsize=(10, 5))
        data = tensor.detach().cpu().numpy()
        data.reshape(data.shape[0], -1)
        data = data.reshape(data_dim)
        plt.imshow(data)
        plt.colorbar()
        plt.title(title)
        if not save:
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_dir, f"{title}.png"))
        plt.close()

    def plot_graph(self, data: torch_geometric.data.Data, title: str = "Graph", save = False):
        plt.figure(figsize=(10, 5))
        plt.scatter(data.x[:, 0], data.x[:, 1], c=data.y, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        if not save:
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_dir, f"{title}.png"))
        plt.close()

    def plot_velocity_field(self, data: torch_geometric.data.Data, 
                          title: str = "Velocity Field", 
                          save: bool = False,
                          show_quiver: bool = True):
        """
        Plot the velocity field from the model's output.
        
        Args:
            data: PyTorch Geometric Data object containing positions and velocities
            title: Title of the plot
            save: Whether to save the plot
            show_quiver: Whether to show velocity vectors
        """
        plt.gca().set_aspect('equal', 'box')
        
        # Extract positions and velocities
        pos = data.pos.detach().cpu().numpy()
        vel = data.x.detach().cpu().numpy()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot ux component
        scatter1 = ax1.scatter(pos[:, 0], pos[:, 1], 
                             c=vel[:, 0], 
                             cmap='viridis',
                             s=10)
        ax1.set_title('Ux Component')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(scatter1, ax=ax1)
        
        # Plot uy component
        scatter2 = ax2.scatter(pos[:, 0], pos[:, 1], 
                             c=vel[:, 1], 
                             cmap='viridis',
                             s=10)
        ax2.set_title('Uy Component')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(scatter2, ax=ax2)
        
        # Add quiver plot if requested
        if show_quiver:
            # Sample points for quiver plot to avoid overcrowding
            sample_size = min(1000, len(pos))
            indices = np.random.choice(len(pos), sample_size, replace=False)
            
            ax1.quiver(pos[indices, 0], pos[indices, 1],
                      vel[indices, 0], np.zeros_like(vel[indices, 0]),
                      color='red', alpha=0.5, scale=20)
            
            ax2.quiver(pos[indices, 0], pos[indices, 1],
                      np.zeros_like(vel[indices, 1]), vel[indices, 1],
                      color='red', alpha=0.5, scale=20)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f"{title}.png"))
        else:
            plt.show()
        plt.close()
