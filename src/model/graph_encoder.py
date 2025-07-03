import torch
from torch_geometric.data import Data
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons

config = commons.get_config('configs/default.yaml')['model']['encoder']

class GraphEncoder(torch.nn.Module):
    def __init__(self, config = config):
        super(GraphEncoder, self).__init__()
        self.config = config
        #conv layers
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])
            
    def forward(self, data: Data, is_verbose: bool = False):
        x = data.x
        x = x.float()  # Ensure input is float32
        for i, (conv, norm) in enumerate(zip(self.convolution_layers.convs, self.convolution_layers.batch_norms)):
            # check if conv take edge_attr as input
            # if is_verbose:
            #     print(f"Convolution layer {i}: {conv.__class__.__name__}, input shape: {x.shape}, edge_index shape: {data.edge_index.shape}, edge_attr shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
            if self.convolution_layers.config['type'] in ['Cheb', 'GCN']:
                x = conv(x = x.float(), edge_index = data.edge_index, edge_weight = data.edge_weight)
            elif self.convolution_layers.config['type'] in ['GAT', 'GMM', 'PNA']:
                x = conv(x = x, edge_index = data.edge_index, edge_attr = data.edge_attr)
            elif self.convolution_layers.config['type'] in ['SAGE']:
                x = conv(x = x, edge_index = data.edge_index)
            if self.convolution_layers.is_skip_connection:
                x = x + data.x
            if i < (len(self.convolution_layers.convs) - 1):
                x = self.convolution_layers.act(x)
            if norm is not None:
                x = norm(x)
            x = self.convolution_layers.dropout(x)
            # print(f"x shape after conv: {x.shape}")
        return x
        
    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

