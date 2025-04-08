import gc
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.autoencoder import Autoencoder
import torch.nn as nn
from torch_geometric.data import Data
from src.utils import commons
config = commons.get_config('configs/default.yaml')

class GAE(nn.Module):
    def __init__(self, config = config):
        super(GAE, self).__init__()
        self.save_config = config
        self.config = self.save_config['model']
        self.encoder = Encoder(self.config['encoder'])
        self.decoder = Decoder(self.config['decoder'])
        self.autoencoder = None
        self.is_autoencoder = self.config['autoencoder']['is_autoencoder']

    def forward(self, data: Data):
        data, pooled_x, pooled_edge_index, pooled_edge_attr, unpool_info = self.encoder(data)
        
        if self.autoencoder is None and self.is_autoencoder:
            self.autoencoder = Autoencoder(config = self.config['autoencoder'], 
                                           input_dim= pooled_x.shape[0] * pooled_x.shape[1])
        
        if not self.is_autoencoder:
            latent_variables = None
            decoded_pooled_x = pooled_x
            del pooled_x
            gc.collect()
        else:
            latent_variables, decoded_pooled_x = self.autoencoder(pooled_x)

        data = self.decoder(data = data, 
                            pooled_x = decoded_pooled_x, 
                            pooled_edge_index = pooled_edge_index, 
                            pooled_edge_attr = pooled_edge_attr, 
                            unpool_info = unpool_info, 
                            unpool_layer = self.encoder.pooling_layer)
        return data.x, latent_variables

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        if self.autoencoder is not None:
            self.autoencoder.reset_parameters()

