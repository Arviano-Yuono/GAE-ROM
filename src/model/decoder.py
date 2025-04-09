import torch
import gc
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
                unpool_layer = None, 
                unpool_info = None):

        if pooled_x is None:
            raise ValueError("Input pooled_x is None. Please ensure pooled_x is not None.")

        if pooled_edge_index is None:
            raise ValueError("Input pooled_edge_index is None. Please ensure pooled_edge_index is not None.")

        if self.is_unpooling and unpool_info is None:
            raise ValueError("Input unpool_info is None. Please ensure unpool_info is not None.")

        if self.is_unpooling and unpool_layer is None:
            raise ValueError("Input unpool is None. Please ensure unpool model is not None.")
        
        self.unpool_info = unpool_info
        self.unpool_layer = unpool_layer

        pooled_x = self.convolution_layers.act(
            self.convolution_layers.convs[0](x = pooled_x, edge_index = pooled_edge_index))
        pooled_x = self.convolution_layers.batch_norms[0](pooled_x)
        pooled_x = self.convolution_layers.dropout(pooled_x)

        if self.is_unpooling:
            data.x, data.edge_index = self.unpool(data, pooled_x, unpool_info)
        else:
            data.x = pooled_x
            data.edge_index = pooled_edge_index
            del pooled_x, pooled_edge_index
            gc.collect()
                
        for conv, norm in zip(self.convolution_layers.convs[1:], self.convolution_layers.batch_norms[1:]):
            data.x = self.convolution_layers.act(conv(x = data.x, edge_index = data.edge_index))
            data.x = norm(data.x)
            data.x = self.convolution_layers.dropout(data.x)
        return data


    def unpool(self, data: Data, pooled_x: torch.Tensor, unpool_info: torch.Tensor):
        if self.unpool_method == 'TopKPool':
            assert unpool_info.shape == data.x.shape
            data.x = torch.zeros_like(data.x, device=data.x.device)
            data.x[unpool_info] = pooled_x   #use perm from topkpool
            return data.x, data.edge_index
        
        elif self.unpool_method == 'EdgePool':
            data.x, data.edge_index, _ = self.unpool_layer.unpool(pooled_x, unpool_info = unpool_info)
            return data.x, data.edge_index
        else:
            raise ValueError("Invalid unpool method: {}".format(self.unpool_method))

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

