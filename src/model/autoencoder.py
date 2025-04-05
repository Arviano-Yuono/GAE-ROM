import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import commons

config = commons.get_config('configs/default.yaml')['model']['autoencoder']

class Autoencoder(nn.Module):
    def __init__(self, config = config):
        super(Autoencoder, self).__init__()
        self.config = config

        self.encoder = nn.Sequential()
        for i in range(len(self.config['encoder_layers'])-1):
            self.encoder.add_module(f'encoder_layer_{i}', nn.Linear(self.config['encoder_layers'][i], self.config['encoder_layers'][i+1]))
            self.encoder.add_module(f'encoder_layer_{i}_relu', nn.ReLU())

        self.decoder = nn.Sequential()
        for i in range(len(self.config['decoder_layers'])-1):
            self.decoder.add_module(f'decoder_layer_{i}', nn.Linear(self.config['decoder_layers'][i], self.config['decoder_layers'][i+1]))
            self.decoder.add_module(f'decoder_layer_{i}_relu', nn.ReLU())

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
