import torch
import gc
from torch_geometric.data import Data
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons

config = commons.get_config('configs/default.yaml')['model']['decoder']

class GraphDecoder(torch.nn.Module):
    def __init__(self, config = config):
        super(GraphDecoder, self).__init__()
        self.config = config
        #conv layers
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])

    def forward(self, data: Data, decoded_reshaped_x: torch.Tensor):
        x = decoded_reshaped_x
        idx = 0
        for conv, norm in zip(self.convolution_layers.convs, self.convolution_layers.batch_norms):
            if self.convolution_layers.config['type'] in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = self.convolution_layers.act(conv(x, data.edge_index, data.edge_weight))
            elif self.convolution_layers.config['type'] in ['GATConv']:
                x = self.convolution_layers.act(conv(x, data.edge_index, data.edge_attr))

            if (idx != len(self.convolution_layers.convs) - 2):
                x = self.convolution_layers.act(x)

            if self.convolution_layers.is_skip_connection:
                x = x + decoded_reshaped_x

            if norm is not None:
                x = norm(x)
            x = self.convolution_layers.dropout(x)
            idx += 1
        return x

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

