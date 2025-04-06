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
