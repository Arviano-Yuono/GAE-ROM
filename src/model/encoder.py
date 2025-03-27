from model.convolution_layers import ConvolutionLayers
from torch_geometric.nn import TopKPooling

from src.utils import commons

config = commons.get_config('configs/default.yaml')

class Encoder:
    def __init__(self, config = config['model']['encoder']):
        self.config = config
        self.convolution_layers = ConvolutionLayers(config)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.convolution_layers(x, edge_index)
        return x
    
    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

