from sklearn import get_config
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.commons import get_activation, get_config

config = get_config('configs/default.yaml')['model']['autoencoder']

class LinearAutoencoder(nn.Module):
    def __init__(self, config = config, input_dim = None):
        super(LinearAutoencoder, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.act_name = self.config['act']
        self.act = get_activation(self.act_name)

        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module(f'encoder_layer_0', nn.Linear(self.input_dim, self.config['encoder_layers'][0]))
        self.encoder.add_module(f'encoder_layer_0_'+self.act_name, self.act)
        for i in range(len(self.config['encoder_layers'])-1):
            self.encoder.add_module(f'encoder_layer_{i+1}', nn.Linear(self.config['encoder_layers'][i], self.config['encoder_layers'][i+1]))
            self.encoder.add_module(f'encoder_layer_{i+1}_'+self.act_name, self.act)
        self.encoder.add_module(f'encoder_layer_latent_dim', nn.Linear(self.config['encoder_layers'][-1], self.config['latent_dim']))
        
        # decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module(f'decoder_layer_latent_dim_'+self.act_name, self.act)
        self.decoder.add_module(f'decoder_layer_latent_dim', nn.Linear(self.config['latent_dim'], self.config['decoder_layers'][0]))
        for i in range(len(self.config['decoder_layers'])-1):
            self.decoder.add_module(f'decoder_layer_{i}', nn.Linear(self.config['decoder_layers'][i], self.config['decoder_layers'][i+1]))
            self.decoder.add_module(f'decoder_layer_{i}_'+self.act_name, self.act)
        self.decoder.add_module(f'decoder_layer_output', nn.Linear(self.config['decoder_layers'][-1], self.input_dim))
        self.decoder.add_module(f'decoder_layer_output_'+self.act_name, self.act)

    def forward(self, x):
        shape = x.shape
        flattened_x = torch.flatten(x)
        x_encoded = self.encoder(flattened_x)
        x_decoded = self.decoder(x_encoded)
        x_decoded = x_decoded.reshape(shape)
        return x_encoded, x_decoded
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
