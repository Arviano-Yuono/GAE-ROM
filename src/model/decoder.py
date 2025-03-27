from torch_geometric.data import Data
from torch import nn
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons

config = commons.get_config('configs/default.yaml')

class Decoder(nn.Module):
    def __init__(self, config = config):
        super(Decoder, self).__init__()
        self.config = config
        self.input_dim = config['model']['convolution_layers']['dim']
        self.hidden_channels = config['model']['convolution_layers']['hidden_channels'][0]
        self.convolution_layers = ConvolutionLayers(config)

    def forward(self, data: Data):
        if data.x is None:
            raise ValueError("Input data.x is None. Please ensure data has node features.")
            
        data.x = self.convolution_layers.fc2(data.x)
        data.x = self.convolution_layers.act(self.convolution_layers.fc1(data.x))
        data.x = data.x.reshape(data.num_graphs, self.input_dim * self.hidden_channels)
        
        for conv in self.convolution_layers.convs:
            data.x = self.convolution_layers.act(conv(data.x, data.edge_index))
            data.x = self.convolution_layers.dropout(data.x)
        return data

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

