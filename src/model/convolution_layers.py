import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from src.utils import commons

config = commons.get_config('configs/default.yaml')['model']['convolution_layers']

class ConvolutionLayers(nn.Module):
    def __init__(self, config = config):
        super(ConvolutionLayers, self).__init__()
        self.config = config

        self.act = F.relu if self.config['act'] == 'relu' else F.elu
        self.dropout = nn.Dropout(self.config['dropout'])

        self.convs = nn.ModuleList()
        if self.config['type'] == 'GMMConv':
            self.convs.append(gnn.GMMConv(self.config['dim'],  # input channels (velocity x,y)
                                         self.config['hidden_channels'][0],  # output channels
                                         dim=self.config['dim'], 
                                         kernel_size=self.config['kernel_size']))
            
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GMMConv(hidden_channel,  
                                            self.config['hidden_channels'][i+1], 
                                            dim=self.config['dim'], 
                                            kernel_size=self.config['kernel_size']))
                
        elif self.config['type'] == 'ChebConv':
            self.convs.append(gnn.ChebConv(self.config['dim'],  # input channels
                                         self.config['hidden_channels'][0],  # output channels
                                         K=self.config['K']))
            
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.ChebConv(hidden_channel, 
                                             self.config['hidden_channels'][i+1], 
                                             K=self.config['K']))
                
        elif self.config['type'] == 'GCNConv':
            self.convs.append(gnn.GCNConv(self.config['dim'],  # input channels
                                         self.config['hidden_channels'][0]))  # output channels
            
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GCNConv(hidden_channel, 
                                            self.config['hidden_channels'][i+1]))
                
        elif self.config['type'] == 'GATConv':
            self.convs.append(gnn.GATConv(self.config['dim'],  # input channels
                                         self.config['hidden_channels'][0]))  # output channels
            
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GATConv(hidden_channel, 
                                            self.config['hidden_channels'][i+1]))
                
        else:
            raise ValueError(f"Invalid message passing type: {self.config['type']}")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
