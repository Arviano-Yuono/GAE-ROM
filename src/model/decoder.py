import torch
import torch_geometric
from torch_geometric.data import Data
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons

config = commons.get_config('configs/default.yaml')['model']['decoder']

class Decoder(torch.nn.Module):
    def __init__(self, config = config):
        super(Decoder, self).__init__()
        self.config = config
        #conv layers
        self.input_dim = config['dim']
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])
        #unpool layer
        self.is_unpooling = config['unpool']['is_unpooling']
        self.unpool_method = config['unpool']['type']
        self.unpool_info = None

    def forward(self, data: Data, flattened_data: torch.Tensor = None, pooled_edge_attr = None, unpool = None, unpool_info = None):
        self.unpool_info = unpool_info

        if flattened_data is None:
            raise ValueError("Input flattened_data is None. Please ensure data has node features.")

        if unpool_info is None:
            raise ValueError("Input unpool_info is None. Please ensure unpool_info is not None.")

        if self.is_unpooling and unpool is None:
            raise ValueError("Input unpool is None. Please ensure unpool model is not None.")
        
        data.x = flattened_data.reshape(data.x.shape)

        data.x = self.convolution_layers.act(
            self.convolution_layers.convs[0](data.x, data.edge_index, pooled_edge_attr))

        if unpool is not None:
            data.x = unpool.unpool(data.x, unpool_info = unpool_info)

        for conv in self.convolution_layers.convs[1:]:
            data.x = self.convolution_layers.act(conv(data.x, data.edge_index, data.edge_attr))
            data.x = self.convolution_layers.dropout(data.x)
        return data

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

