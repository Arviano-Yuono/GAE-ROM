import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from src.utils import commons

config = commons.get_config('configs/default.yaml')

class ConvolutionLayers(nn.Module):
    def __init__(self, config = config):
        super(ConvolutionLayers, self).__init__()
        self.config = config

        self.act = F.relu if self.config['model']['convolution_layers']['act'] == 'relu' else F.elu
        self.dropout = nn.Dropout(self.config['model']['convolution_layers']['dropout'])

        self.convs = nn.ModuleList()
        for hidden_channel in self.config['model']['convolution_layers']['hidden_channels']:
            if self.config['model']['convolution_layers']['type'] == 'GMMConv':
                self.convs.append(gnn.GMMConv(hidden_channel,  
                                              hidden_channel, 
                                              dim=self.config['model']['convolution_layers']['dim'], 
                                              kernel_size=self.config['model']['convolution_layers']['kernel_size']))
                
            elif self.config['model']['convolution_layers']['type'] == 'ChebConv':
                self.convs.append(gnn.ChebConv(hidden_channel, 
                                               hidden_channel, 
                                               K=self.config['model']['convolution_layers']['K']))
                
            elif self.config['model']['convolution_layers']['type'] == 'GCNConv':
                self.convs.append(gnn.GCNConv(hidden_channel, 
                                              hidden_channel))
                
            elif self.config['model']['convolution_layers']['type'] == 'GATConv':
                self.convs.append(gnn.GATConv(hidden_channel, 
                                              hidden_channel))
                
            else:
                raise ValueError(f"Invalid message passing type: {self.config['model']['convolution_layers']['type']}")

        self.fc1 = nn.Linear(self.config['model']['convolution_layers']['hidden_channels'], self.config['model']['convolution_layers']['ffn'])
        self.fc2 = nn.Linear(self.config['model']['convolution_layers']['ffn'], self.config['model']['convolution_layers']['bottleneck'])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
