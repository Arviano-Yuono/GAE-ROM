from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from src.utils.commons import get_activation_function

class ConvolutionLayers(nn.Module):
    def __init__(self, config):
        super(ConvolutionLayers, self).__init__()
        self.config = config
        self.act_name = config['act']
        self.act = get_activation_function(self.act_name)
        self.dropout = nn.Dropout(self.config['dropout'])
        self.is_skip_connection = self.config['is_skip_connection']
        self.convs = nn.ModuleList()

        if self.config['is_batch_norm']:
            self.batch_norms = nn.ModuleList()
        else:
            self.batch_norms = []

        if self.config['type'] == 'GMM':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.conv.GMMConv(in_channels=hidden_channel,  
                                            out_channels=self.config['hidden_channels'][i+1], 
                                            dim=self.config['dim'], 
                                            kernel_size=self.config['kernel_size']))
                if self.config['is_batch_norm']:
                    self.batch_norms.append(nn.BatchNorm1d(self.config['hidden_channels'][i+1]))
                else:
                    self.batch_norms.append(None)
        elif self.config['type'] == 'SAGE':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.conv.SAGEConv(in_channels=hidden_channel, 
                                               out_channels=self.config['hidden_channels'][i+1],
                                               normalize=False))
                if self.config['is_batch_norm']:
                    self.batch_norms.append(nn.BatchNorm1d(self.config['hidden_channels'][i+1]))
                else:
                    self.batch_norms.append(None)

        elif self.config['type'] == 'Cheb':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.ChebConv(hidden_channel, 
                                             self.config['hidden_channels'][i+1], 
                                             K=self.config['K']))
                if self.config['is_batch_norm']:
                    self.batch_norms.append(nn.BatchNorm1d(self.config['hidden_channels'][i+1]))
                else:
                    self.batch_norms.append(None)

        elif self.config['type'] == 'GCN':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                self.convs.append(gnn.conv.GCNConv(hidden_channel, 
                                            self.config['hidden_channels'][i+1],
                                            normalize=False))
                if self.config['is_batch_norm']:
                    self.batch_norms.append(nn.BatchNorm1d(self.config['hidden_channels'][i+1]))
                else:
                    self.batch_norms.append(None)

        elif self.config['type'] == 'GAT':
            for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
                out_dim = self.config['hidden_channels'][i+1]
                is_last = (i == len(self.config['hidden_channels']) - 2)
                if is_last or out_dim < self.config['head']:
                    self.convs.append(gnn.conv.GATv2Conv(
                        in_channels=hidden_channel,
                        out_channels=out_dim,
                        heads=1,
                        dropout=self.config['dropout']
                    ))
                    if self.config['is_batch_norm']:
                        self.batch_norms.append(nn.BatchNorm1d(out_dim))
                    else:
                        self.batch_norms.append(None)
                else:
                    assert out_dim % self.config['head'] == 0, \
                        f"GAT: hidden_channels[{i+1}] = {out_dim} not divisible by head = {self.config['head']}"

                    out_per_head = out_dim // self.config['head']
                    self.convs.append(gnn.conv.GATv2Conv(
                        in_channels=hidden_channel,
                        out_channels=out_per_head,
                        heads=self.config['head'],
                        dropout=self.config['dropout']
                    ))
                    if self.config['is_batch_norm']:
                        self.batch_norms.append(nn.BatchNorm1d(out_dim))
                    else:
                        self.batch_norms.append(None)
        else:
            raise ValueError(f"Invalid message passing type: {self.config['type']}")
        self.reset_parameters()

    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_uniform_(param)

        for batch_norm in self.batch_norms:
            if batch_norm is not None:
                batch_norm.reset_parameters()

