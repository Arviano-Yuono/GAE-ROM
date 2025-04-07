import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import commons

config = commons.get_config('configs/default.yaml')['model']['autoencoder']

class Autoencoder(nn.Module):
    def __init__(self, config = config, input_dim = None):
        super(Autoencoder, self).__init__()
        self.config = config
        if input_dim is None:
            raise ValueError("input_dim must be provided")
        self.input_dim = input_dim
        # print(f"input_dim: {self.input_dim}")
        # self.input_dim = 279088
        # self.input_dim = 493536
        
        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module(f'encoder_layer_0', nn.Linear(self.input_dim, self.config['encoder_layers'][0]))
        self.encoder.add_module(f'encoder_layer_0_relu', nn.ReLU())

        for i in range(len(self.config['encoder_layers'])-1):
            self.encoder.add_module(f'encoder_layer_{i+1}', nn.Linear(self.config['encoder_layers'][i], self.config['encoder_layers'][i+1]))
            self.encoder.add_module(f'encoder_layer_{i+1}_relu', nn.ReLU())
        self.encoder.add_module(f'encoder_layer_latent_dim', nn.Linear(self.config['encoder_layers'][-1], self.config['latent_dim']))
        self.encoder.add_module(f'encoder_layer_latent_dim_relu', nn.ReLU())
        
        # decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module(f'decoder_layer_latent_dim', nn.Linear(self.config['latent_dim'], self.config['decoder_layers'][0]))
        self.decoder.add_module(f'decoder_layer_latent_dim_relu', nn.ReLU())
        for i in range(len(self.config['decoder_layers'])-1):
            self.decoder.add_module(f'decoder_layer_{i}', nn.Linear(self.config['decoder_layers'][i], self.config['decoder_layers'][i+1]))
            self.decoder.add_module(f'decoder_layer_{i}_relu', nn.ReLU())
        self.decoder.add_module(f'decoder_layer_output', nn.Linear(self.config['decoder_layers'][-1], self.input_dim))
        self.decoder.add_module(f'decoder_layer_output_relu', nn.ReLU())

    def forward(self, x):
        shape = x.shape
        flattened_x = torch.flatten(x)
        x_encoded = self.encoder(flattened_x)
        x_decoded = self.decoder(x_encoded)
        x_decoded = x_decoded.reshape(shape)
        # print(f"x_input.shape: {shape}")
        # print(f"x_decoded.shape: {x_decoded.shape}")
        return x_encoded, x_decoded
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
