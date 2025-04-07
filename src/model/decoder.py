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
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])
        #unpool layer
        self.is_unpooling = config['unpool']['is_unpooling']
        self.unpool_method = config['unpool']['type']
        self.unpool_info = None

    def forward(self, data: Data, 
                pooled_x: torch.Tensor = None, 
                pooled_edge_index = None, 
                pooled_edge_attr = None, 
                unpool = None, 
                unpool_info = None):
        self.unpool_info = unpool_info

        if pooled_x is None:
            raise ValueError("Input pooled_x is None. Please ensure pooled_x is not None.")

        if pooled_edge_index is None:
            raise ValueError("Input pooled_edge_index is None. Please ensure pooled_edge_index is not None.")

        if unpool_info is None:
            raise ValueError("Input unpool_info is None. Please ensure unpool_info is not None.")

        if self.is_unpooling and unpool is None:
            raise ValueError("Input unpool is None. Please ensure unpool model is not None.")
        
        pooled_x = self.convolution_layers.act(
            self.convolution_layers.convs[0](pooled_x, pooled_edge_index))
        # data.x = self.convolution_layers.act(
        #     self.convolution_layers.convs[0](data.x, data.edge_index, data.edge_attr))

        if unpool is not None:
            if self.unpool_method == 'TopKPool':
                data.x = torch.zeros_like(data.x, device=data.x.device)
                data.x[unpool_info] = pooled_x   #use perm from topkpool
            elif self.unpool_method == 'EdgePool':
                data.x, data.edge_index, _ = unpool.unpool(pooled_x, unpool_info = unpool_info)
            else:
                data.x = pooled_x
                data.edge_index = pooled_edge_index
        for i in range(1, len(self.convolution_layers.convs)):
            conv = self.convolution_layers.convs[i]
            data.x = self.convolution_layers.act(conv(data.x, data.edge_index))
            # data.x = self.convolution_layers.act(conv(data.x, data.edge_index, data.edge_attr))
            data.x = self.convolution_layers.dropout(data.x)
        return data

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

