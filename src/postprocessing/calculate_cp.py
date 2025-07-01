import torch
import torch_geometric

def calculate_cp(data: torch_geometric.data.Data,
                 surface_mask: torch.Tensor,
                 p_inf: float,
                 q_inf: float):
    
    p = data.x[:, 0]
    rho = data.x[:, 1]
    u = data.x[:, 2]