import torch_geometric
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.autoencoder import Autoencoder
from src.model.convolution_layers import ConvolutionLayers
import torch.nn as nn
from torch_geometric.data import Data
from src.utils import commons
config = commons.get_config('configs/default.yaml')['model']

class GAE(nn.Module):
    def __init__(self, num_nodes, config = config):
        super(GAE, self).__init__()
        self.num_nodes = num_nodes
        self.flattened_dim = int(num_nodes * config['encoder']['pool']['ratio'] * config['encoder']['convolution_layers']['hidden_channels'][-1])
        #models
        # implement torch_geometric profiler
        # self.profiler = torch_geometric.profile.Profiler(self)
        # self.profiler.start()
        self.encoder = Encoder(config['encoder'])
        self.decoder = Decoder(config['decoder'])
        self.autoencoder = Autoencoder(config = config['autoencoder'], input_dim= self.flattened_dim)
        # self.profiler.stop()
        
    def forward(self, data: Data):
        data, flattened_data, unpool_info, pooled_edge_attr = self.encoder(data)
        latent_variables, data.x = self.autoencoder(flattened_data)
        data = self.decoder(data = data, flattened_data = flattened_data, pooled_edge_attr = pooled_edge_attr, unpool_info = unpool_info, unpool = self.encoder.pool)
        return data.x, latent_variables

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.autoencoder.reset_parameters()

