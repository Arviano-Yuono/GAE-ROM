from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.autoencoder import Autoencoder
from src.model.convolution_layers import ConvolutionLayers
import torch.nn as nn
from torch_geometric.data import Data
from src.utils import commons
config = commons.get_config('configs/default.yaml')['model']

class GAE(nn.Module):
    def __init__(self, config = config):
        super(GAE, self).__init__()
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])
        self.encoder = Encoder(self.convolution_layers, config['encoder'])
        self.decoder = Decoder(self.convolution_layers, config['decoder'])
        self.autoencoder = Autoencoder(config['autoencoder'])

    def forward(self, data: Data):
        data = self.encoder(data)
        latent_variables, data.x = self.autoencoder(data.x)
        data = self.decoder(data)
        return data.x, latent_variables

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.autoencoder.reset_parameters()

