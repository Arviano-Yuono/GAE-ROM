import torch_geometric
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.autoencoder import Autoencoder
import torch.nn as nn
from torch_geometric.data import Data
from src.utils import commons
config = commons.get_config('configs/default.yaml')['model']

class GAE(nn.Module):
    def __init__(self, config = config):
        super(GAE, self).__init__()
        self.latent_variables = None
        self.encoder = Encoder(config['encoder'])
        self.decoder = Decoder(config['decoder'])
        self.autoencoder = None
        
    def forward(self, data: Data):
        data, pooled_x, pooled_edge_index, pooled_edge_attr, unpool_info = self.encoder(data)
        if self.autoencoder is None:
            self.autoencoder = Autoencoder(config = config['autoencoder'], input_dim= pooled_x.shape[0] * pooled_x.shape[1])
        self.latent_variables, pooled_x = self.autoencoder(pooled_x)
        data = self.decoder(data = data, 
                            pooled_x = pooled_x, 
                            pooled_edge_index = pooled_edge_index, 
                            pooled_edge_attr = pooled_edge_attr, 
                            unpool_info = unpool_info, 
                            unpool = self.encoder.pooling_layer)
        return data.x, self.latent_variables

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        if self.autoencoder is not None:
            self.autoencoder.reset_parameters()

