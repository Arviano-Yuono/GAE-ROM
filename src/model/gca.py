import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class GCA(nn.Module):
    def __init__():
        pass
    def encoder(self, data):
        pass
    def decoder(self, x, data):
        pass
    def reset_parameters(self):
        pass
    def forward(self, data):
        pass

class Encoder(GCA):
    def __init__(self, hidden_channels, bottleneck, input_size, ffn, skip, act=F.elu, conv='GMMConv'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = len(self.hidden_channels)
        self.act = act
        self.ffn = ffn
        self.skip = skip
        self.bottleneck = bottleneck
        self.input_size = input_size
        self.conv = conv

        self.fc_out1 = nn.Linear(self.input_size, self.ffn)
        self.fc_out2 = nn.Linear(self.ffn, self.bottleneck)

        self.down_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            if self.conv=='GMMConv':
                self.down_convs.append(gnn.GMMConv(self.hidden_channels[i], self.hidden_channels[i+1], dim=1, kernel_size=5))
            elif self.conv=='ChebConv':
                self.down_convs.append(gnn.ChebConv(self.hidden_channels[i], self.hidden_channels[i+1], K=5))
            elif self.conv=='GCNConv':
                self.down_convs.append(gnn.GCNConv(self.hidden_channels[i], self.hidden_channels[i+1]))
            elif self.conv=='GATConv':
                self.down_convs.append(gnn.GATConv(self.hidden_channels[i], self.hidden_channels[i+1]))
            else:
                raise NotImplementedError('Invalid convolution selected. Please select one of [GMMConv, ChebConv, GCNConv, GATConv]')
            
        self.reset_parameters()

    def encoder(self, data):
        x = data.x
        for layer in self.down_convs:
            if self.conv in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = layer(x, data.edge_index, data.edge_weight)
            elif self.conv in ['GATConv']:
                x = layer(x, data.edge_index, data.edge_attr)
            x = self.act(x)
        x = x.reshape(data.num_graphs*self.hidden_channels[-1], self.input_size)
        x = self.act(self.fc_out1(x))
        x = self.act(self.fc_out2(x))
        return x

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()

class Decoder(GCA):
    def __init__(self, hidden_channels, bottleneck, input_size, ffn, skip, act=F.elu, conv='GMMConv'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = len(self.hidden_channels)
        self.act = act
        self.ffn = ffn
        self.skip = skip
        self.bottleneck = bottleneck
        self.input_size = input_size
        self.conv = conv

        self.fc_out1 = nn.Linear(self.bottleneck, self.ffn)
        self.fc_out2 = nn.Linear(self.ffn, self.input_size * self.hidden_channels[-1])

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            if self.conv=='GMMConv':
                self.up_convs.append(gnn.GMMConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2], dim=1, kernel_size=5))
            elif self.conv=='ChebConv':
                self.up_convs.append(gnn.ChebConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2], K=5))
            elif self.conv=='GCNConv':
                self.up_convs.append(gnn.GCNConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2]))
            elif self.conv=='GATConv':
                self.up_convs.append(gnn.GATConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2]))
            else:
                raise NotImplementedError('Invalid convolution selected. Please select one of [GMMConv, ChebConv, GCNConv, GATConv]')
            
        
        self.reset_parameters()
    
    def decoder(self, x, data):
        x = self.act(self.fc_out1(x))
        x = self.act(self.fc_out2(x))
        h = x.reshape(data.num_graphs*self.input_size, self.hidden_channels[-1])
        x = h
        idx = 0
        for layer in self.up_convs:
            if self.conv in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = layer(x, data.edge_index, data.edge_weight)
            elif self.conv in ['GATConv']:
                x = layer(x, data.edge_index, data.edge_attr)
            if (idx != self.depth - 2):
                x = self.act(x)
            if self.skip:
                x = x + h
            idx += 1
        return x
    