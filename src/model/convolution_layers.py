from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import LayerNorm, BatchNorm
from torch.nn import Identity
import torch
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
        self.norm_type = config.get('norm_type', 'layer')

        if self.config['is_batch_norm']:
            self.batch_norms = nn.ModuleList()
        else:
            self.batch_norms = []

        def append_norm_layer(i, out_dim):
            if not self.config['is_batch_norm']:
                self.batch_norms.append(None)
                return
            if self.norm_type == 'batch':
                self.batch_norms.append(BatchNorm(out_dim))
            elif self.norm_type == 'layer':
                self.batch_norms.append(LayerNorm(out_dim, affine=True))
            else:
                self.batch_norms.append(Identity())

        conv_type = self.config['type']

        for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
            out_dim = self.config['hidden_channels'][i+1]

            if conv_type == 'GMM':
                self.convs.append(gnn.GMMConv(hidden_channel, out_dim, dim=self.config['dim'], kernel_size=self.config['kernel_size']))
            elif conv_type == 'SAGE':
                self.convs.append(gnn.SAGEConv(hidden_channel, out_dim, normalize=False))
            elif conv_type == 'Cheb':
                self.convs.append(gnn.ChebConv(hidden_channel, out_dim, K=self.config['K']))
            elif conv_type == 'GCN':
                self.convs.append(gnn.GCNConv(hidden_channel, out_dim, normalize=False))
            elif conv_type == 'PNA':
                self.convs.append(gnn.PNAConv(hidden_channel, out_dim, 
                                              aggregators=self.config['aggregators'], 
                                              scalers=self.config['scalers'], 
                                              deg=torch.tensor(self.config['deg']), 
                                              edge_dim=2))
            elif conv_type == 'GAT':
                is_last = (i == len(self.config['hidden_channels']) - 2)
                heads = self.config.get('head', 1)
                concat = self.config.get('concat', True)
                if is_last or out_dim < heads:
                    self.convs.append(gnn.GATv2Conv(hidden_channel, out_dim, heads=1, dropout=self.config['dropout'], concat=False))
                else:
                    assert out_dim % heads == 0, f"GAT: hidden_channels[{i+1}] = {out_dim} not divisible by head = {heads}"
                    self.convs.append(gnn.GATv2Conv(hidden_channel, out_dim // heads, heads=heads, dropout=self.config['dropout'], concat=True))
            else:
                raise ValueError(f"Invalid message passing type: {conv_type}")

            append_norm_layer(i, out_dim)

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


# from torch import nn
# import torch.nn.functional as F
# import torch_geometric.nn as gnn
# from torch_geometric.nn import LayerNorm, BatchNorm
# import torch
# from src.utils.commons import get_activation_function

# class ConvolutionLayers(nn.Module):
#     def __init__(self, config):
#         super(ConvolutionLayers, self).__init__()
#         self.config = config
#         self.act_name = config['act']
#         self.act = get_activation_function(self.act_name)
#         self.dropout = nn.Dropout(self.config['dropout'])
#         self.is_skip_connection = self.config['is_skip_connection']
#         self.convs = nn.ModuleList()
#         self.norm_type = config.get('norm_type', 'layer')  # default to 'layer'

#         if self.config['is_batch_norm']:
#             self.batch_norms = nn.ModuleList()
#         else:
#             self.batch_norms = []

#         if self.config['type'] == 'GMM':
#             for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
#                 self.convs.append(gnn.conv.GMMConv(in_channels=hidden_channel,  
#                                             out_channels=self.config['hidden_channels'][i+1], 
#                                             dim=self.config['dim'], 
#                                             kernel_size=self.config['kernel_size']))
#                 if self.config['is_batch_norm']:
#                     norm_dim = self.config['hidden_channels'][i+1]
#                     if self.norm_type == 'batch' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(nn.BatchNorm1d(norm_dim))
#                     elif self.norm_type == 'layer' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(LayerNorm(norm_dim, affine=True))
#                 else:
#                     self.batch_norms.append(None)

#         elif self.config['type'] == 'SAGE':
#             for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
#                 self.convs.append(gnn.conv.SAGEConv(in_channels=hidden_channel, 
#                                                out_channels=self.config['hidden_channels'][i+1],
#                                                normalize=False))
#                 if self.config['is_batch_norm']:
#                     norm_dim = self.config['hidden_channels'][i+1]
#                     if self.norm_type == 'batch' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(nn.BatchNorm1d(norm_dim))
#                     elif self.norm_type == 'layer' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(LayerNorm(norm_dim, affine=True))
#                     elif self.norm_type == 'layer' and i >= len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(Identity())
#                 else:
#                     self.batch_norms.append(None)


#         elif self.config['type'] == 'Cheb':
#             for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
#                 self.convs.append(gnn.ChebConv(hidden_channel, 
#                                              self.config['hidden_channels'][i+1], 
#                                              K=self.config['K']))
#                 if self.config['is_batch_norm']:
#                     norm_dim = self.config['hidden_channels'][i+1]
#                     if self.norm_type == 'batch' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(nn.BatchNorm1d(norm_dim))
#                     elif self.norm_type == 'layer' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(LayerNorm(norm_dim, affine=True))
#                     elif self.norm_type == 'layer' and i >= len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(Identity())
#                 else:
#                     self.batch_norms.append(None)


#         elif self.config['type'] == 'GCN':
#             for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
#                 self.convs.append(gnn.conv.GCNConv(hidden_channel, 
#                                             self.config['hidden_channels'][i+1],
#                                             normalize=False))
#                 if self.config['is_batch_norm']:
#                     norm_dim = self.config['hidden_channels'][i+1]
#                     if self.norm_type == 'batch' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(nn.BatchNorm1d(norm_dim))
#                     elif self.norm_type == 'layer' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(LayerNorm(norm_dim, affine=True))
#                     elif self.norm_type == 'layer' and i >= len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(Identity())
#                 else:
#                     self.batch_norms.append(None)


#         elif self.config['type'] == 'PNA':
#             for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
#                 self.convs.append(gnn.conv.PNAConv(
#                     in_channels=hidden_channel,
#                     out_channels=self.config['hidden_channels'][i+1],
#                     aggregators=self.config['aggregators'],
#                     scalers=self.config['scalers'],
#                     deg=torch.tensor(self.config['deg']),
#                     edge_dim=2
#                 ))
#                 if self.config['is_batch_norm']:
#                     norm_dim = self.config['hidden_channels'][i+1]
#                     if self.norm_type == 'batch' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(nn.BatchNorm1d(norm_dim))
#                     elif self.norm_type == 'layer' and i < len(self.config['hidden_channels']) - 2:
#                         self.batch_norms.append(LayerNorm(norm_dim, affine=True))
#                 else:
#                     self.batch_norms.append(None)


#         elif self.config['type'] == 'GAT':
#             for i, hidden_channel in enumerate(self.config['hidden_channels'][:-1]):
#                 out_dim = self.config['hidden_channels'][i+1]
#                 is_last = (i == len(self.config['hidden_channels']) - 2)
#                 heads = self.config.get('heads', 1)
#                 concat = self.config.get('concat', True)
#                 if is_last or out_dim < self.config['head']:
#                     self.convs.append(gnn.conv.GATv2Conv(
#                         in_channels=hidden_channel,
#                         out_channels=out_dim // heads if concat else out_dim,
#                         heads=1,
#                         dropout=self.config['dropout'],
#                         concat=False
#                     ))
#                     if self.config['is_batch_norm']:
#                         norm_dim = self.config['hidden_channels'][i+1]
#                         if self.norm_type == 'batch' and i < len(self.config['hidden_channels']) - 2:
#                             self.batch_norms.append(nn.BatchNorm1d(norm_dim))
#                         elif self.norm_type == 'layer' and i < len(self.config['hidden_channels']) - 2:
#                             self.batch_norms.append(LayerNorm(norm_dim, affine=True))
#                         else:
#                             raise ValueError(f"Unknown norm_type: {self.norm_type}")
#                     else:
#                         self.batch_norms.append(None)

#                 else:
#                     assert out_dim % self.config['head'] == 0, \
#                         f"GAT: hidden_channels[{i+1}] = {out_dim} not divisible by head = {self.config['head']}"

#                     out_per_head = out_dim // self.config['head']
#                     self.convs.append(gnn.conv.GATv2Conv(
#                         in_channels=hidden_channel,
#                         out_channels=out_per_head,
#                         heads=self.config['head'],
#                         dropout=self.config['dropout']
#                     ))
#                     if self.config['is_batch_norm']:
#                         norm_dim = self.config['hidden_channels'][i+1]
#                         if self.norm_type == 'batch' and i < len(self.config['hidden_channels']) - 2:
#                             self.batch_norms.append(nn.BatchNorm1d(norm_dim))
#                         elif self.norm_type == 'layer' and i < len(self.config['hidden_channels']) - 2:
#                             self.batch_norms.append(LayerNorm(norm_dim, affine=True))
#                         else:
#                             raise ValueError(f"Unknown norm_type: {self.norm_type}")
#                     else:
#                         self.batch_norms.append(None)

#         else:
#             raise ValueError(f"Invalid message passing type: {self.config['type']}")
#         self.reset_parameters()

    
#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#             for name, param in conv.named_parameters():
#                 if 'bias' in name:
#                     nn.init.constant_(param, 0)
#                 else:
#                     nn.init.kaiming_uniform_(param)

#         for batch_norm in self.batch_norms:
#             if batch_norm is not None:
#                 batch_norm.reset_parameters()

