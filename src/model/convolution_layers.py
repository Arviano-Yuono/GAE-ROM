from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class ConvolutionLayers(nn.Module):
    def __init__(self, config):
        super(ConvolutionLayers, self).__init__()
        self.config = config

        if self.config['act'] == 'relu':
            self.act = F.relu
        elif self.config['act'] == 'elu':
            self.act = F.elu
        elif self.config['act'] == 'leaky_relu':
            self.act = F.leaky_relu
        elif self.config['act'] == 'tanh':
            self.act = F.tanh
        elif self.config['act'] == 'sigmoid':
            self.act = F.sigmoid
        elif self.config['act'] == 'softplus':
            self.act = F.softplus
        else:
            raise ValueError(f"Invalid activation function: {self.config['act']}")

        self.dropout = nn.Dropout(self.config['dropout'])

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        if self.config['type'] == 'GMMConv':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GMMConv(in_channels=hidden_channel,  
                                            out_channels=self.config['hidden_channels'][i+1], 
                                            dim=self.config['dim'], 
                                            kernel_size=self.config['kernel_size'],
                                            dropout=self.config['dropout']))
                self.batch_norms.append(nn.BatchNorm1d(self.config['hidden_channels'][i+1]))

        elif self.config['type'] == 'GraphSAGE':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GraphSAGE(in_channels=hidden_channel, 
                                               hidden_channels=self.config['hidden_channels'][i+1],
                                               dropout=self.config['dropout'],
                                               num_layers=self.config['num_layers'],
                                               out_channels=self.config['hidden_channels'][i+1]))
                self.batch_norms.append(nn.BatchNorm1d(self.config['hidden_channels'][i+1]))

        elif self.config['type'] == 'ChebConv':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.ChebConv(hidden_channel, 
                                             self.config['hidden_channels'][i+1], 
                                             K=self.config['K']))
                self.batch_norms.append(nn.BatchNorm1d(self.config['hidden_channels'][i+1]))

        elif self.config['type'] == 'GCNConv':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.GCNConv(hidden_channel, 
                                            self.config['hidden_channels'][i+1],
                                            normalize=False))
                self.batch_norms.append(nn.BatchNorm1d(self.config['hidden_channels'][i+1]))

        elif self.config['type'] == 'GATConv':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                out_dim = self.config['hidden_channels'][i+1]
                is_last = (i == len(self.config['hidden_channels']) - 2)
                if is_last or out_dim < self.config['head']:
                    self.convs.append(gnn.GATv2Conv(
                        in_channels=hidden_channel,
                        out_channels=out_dim,
                        heads=1,
                        dropout=self.config['dropout']
                    ))
                    self.batch_norms.append(nn.BatchNorm1d(out_dim))
                else:
                    assert out_dim % self.config['head'] == 0, \
                        f"GAT: hidden_channels[{i+1}] = {out_dim} not divisible by head = {self.config['head']}"

                    out_per_head = out_dim // self.config['head']
                    self.convs.append(gnn.GATv2Conv(
                        in_channels=hidden_channel,
                        out_channels=out_per_head,
                        heads=self.config['head'],
                        dropout=self.config['dropout']
                    ))
                    self.batch_norms.append(nn.BatchNorm1d(out_dim))
        else:
            raise ValueError(f"Invalid message passing type: {self.config['type']}")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

