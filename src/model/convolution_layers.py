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
        # First layer needs to handle input dimension
        if self.config['type'] == 'GMMConv':
            self.convs.append(gnn.GMMConv(2,  # input channels (velocity x,y)
                                         self.config['hidden_channels'][0],  # output channels
                                         dim=self.config['dim'], 
                                         kernel_size=self.config['kernel_size']))
            
            # Add remaining layers
            for hidden_channel in self.config['hidden_channels'][:-1]:
                self.convs.append(gnn.GMMConv(hidden_channel,  
                                            hidden_channel, 
                                            dim=self.config['dim'], 
                                            kernel_size=self.config['kernel_size']))
                
        elif self.config['type'] == 'ChebConv':
            self.convs.append(gnn.ChebConv(2,  # input channels
                                         self.config['hidden_channels'][0],  # output channels
                                         K=self.config['K']))
            
            for hidden_channel in self.config['hidden_channels'][:-1]:
                self.convs.append(gnn.ChebConv(hidden_channel, 
                                             hidden_channel, 
                                             K=self.config['K']))
                
        elif self.config['type'] == 'GCNConv':
            self.convs.append(gnn.GCNConv(2,  # input channels
                                         self.config['hidden_channels'][0]))  # output channels
            
            for hidden_channel in self.config['hidden_channels'][:-1]:
                self.convs.append(gnn.GCNConv(hidden_channel, 
                                            hidden_channel))
                
        elif self.config['type'] == 'GATConv':
            self.convs.append(gnn.GATConv(2,  # input channels
                                         self.config['hidden_channels'][0]))  # output channels
            
            for hidden_channel in self.config['hidden_channels'][:-1]:
                self.convs.append(gnn.GATConv(hidden_channel, 
                                            hidden_channel))
                
        else:
            raise ValueError(f"Invalid message passing type: {self.config['type']}")

        self.fc1 = nn.Linear(self.config['hidden_channels'][-1], self.config['ffn'])
        self.fc2 = nn.Linear(self.config['ffn'], self.config['bottleneck'])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
