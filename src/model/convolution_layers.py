import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from src.utils import commons


class ConvolutionLayers(nn.Module):
    def __init__(self, config):
        super(ConvolutionLayers, self).__init__()
        self.config = config

        self.act = F.relu if self.config['act'] == 'relu' else F.elu
        self.dropout = nn.Dropout(self.config['dropout'])

        self.convs = nn.ModuleList()
        if self.config['type'] == 'GMMConv':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GMMConv(hidden_channel,  
                                            self.config['hidden_channels'][i+1], 
                                            dim=self.config['dim'], 
                                            kernel_size=self.config['kernel_size']))
                
        elif self.config['type'] == 'ChebConv':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.ChebConv(hidden_channel, 
                                             self.config['hidden_channels'][i+1], 
                                             K=self.config['K']))
                
        elif self.config['type'] == 'GCNConv':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GCNConv(hidden_channel, 
                                            self.config['hidden_channels'][i+1]))
                
        elif self.config['type'] == 'GATConv':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GATConv(hidden_channel, 
                                            self.config['hidden_channels'][i+1]))
                
        else:
            raise ValueError(f"Invalid message passing type: {self.config['type']}")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
