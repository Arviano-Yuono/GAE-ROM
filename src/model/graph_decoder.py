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
        for i, (conv, norm) in enumerate(zip(self.convolution_layers.convs, self.convolution_layers.batch_norms)):
            if self.convolution_layers.config['type'] in ['GMM', 'Cheb', 'GCN']:
                x = conv(x, data.edge_index, data.edge_weight)
            elif self.convolution_layers.config['type'] in ['GAT']:
                x = conv(x, data.edge_index, data.edge_attr)
            elif self.convolution_layers.config['type'] in ['SAGE']:
                x = conv(x, data.edge_index)

            if (i != len(self.convolution_layers.convs) - 2):
                x = self.convolution_layers.act(x)

            if self.convolution_layers.is_skip_connection:
                x = x + decoded_reshaped_x

            if norm is not None:
                x = norm(x)
            x = self.convolution_layers.dropout(x)
        return x

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

