import torch
import gc
from torch_geometric.data import Data
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons
import torch.nn as nn

class GraphDecoder(torch.nn.Module):
    def __init__(self, config, device=None):
        super(GraphDecoder, self).__init__()
        self.config = config
        self.device = device
        
        #conv layers
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])

    def forward(self, data: Data, latent_variables: torch.Tensor):
        """
        Forward pass through linear decoder and graph decoder.
        
        Args:
            data: Input graph data (for edge information)
            latent_variables: Latent variables to decode
            
        Returns:
            Decoded/reconstructed features
        """
        # Graph decoding
        if self.config['is_classifier']:
            x = latent_variables.view(-1).unsqueeze(0).expand(data.x.shape[0], -1)  # shape: [num_nodes, latent_var_size]
        else:
            x = latent_variables
        idx = 0
        for conv, norm in zip(self.convolution_layers.convs, self.convolution_layers.batch_norms):
            if self.convolution_layers.config['type'] in ['ChebConv', 'GCNConv']:
                x = conv(x, data.edge_index, data.edge_weight)
            elif self.convolution_layers.config['type'] in ['GATConv', 'GMMConv']:
                x = conv(x, data.edge_index, data.edge_attr)
            elif self.convolution_layers.config['type'] in ['SAGE']:
                x = conv(x, data.edge_index)
            if (idx != len(self.convolution_layers.convs) - 2):
                x = self.convolution_layers.act(x)

            if self.convolution_layers.is_skip_connection:
                x = x + latent_variables

            if norm is not None:
                x = norm(x)
            x = self.convolution_layers.dropout(x)
            idx += 1
        return x

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()


