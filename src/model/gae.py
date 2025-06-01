import gc
from src.model.graph_encoder import GraphEncoder
from src.model.graph_decoder import GraphDecoder
from src.model.autoencoder import LinearAutoencoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from src.utils.commons import get_activation_function, get_config
config = get_config('configs/default.yaml')

class GAE(nn.Module):
    def __init__(self, config = config, num_graphs = None):
        super(GAE, self).__init__()
        self.save_config = config
        self.config = self.save_config['model']

        # graph encoder
        self.graph_encoder = GraphEncoder(self.config['encoder'])

        # graph decoder
        self.graph_decoder = GraphDecoder(self.config['decoder'])

        # autoencoder
        self.is_autoencoder = self.config['autoencoder']['is_autoencoder']
        self.linear_autoencoder = LinearAutoencoder(config = self.config['autoencoder'], 
                                           input_dim= num_graphs * self.config['encoder']['convolution_layers']['hidden_channels'][-1])

        # mapping to vector
        self.maptovec = nn.ModuleList()
        self.act_map = get_activation_function(self.config['maptovec']['act'])
        for k in range(len(self.config['maptovec']['layer_vec'])-1):
            self.maptovec.append(nn.Linear(self.config['maptovec']['layer_vec'][k], self.config['maptovec']['layer_vec'][k+1]))
        
        self.skip_proj = None # for skip connection
        # self.is_skip_connection = self.config['autoencoder']['is_skip_connection']

    def mapping(self, x):
        # Ensure input is float32
        x = x.float()
        idx = 0
        for layer in self.maptovec:
            if (idx==len(self.maptovec)-1): x = layer(x) # last layer doesnt need activ func because its the latent var
            else: x = self.act_map(layer(x))
            idx += 1
        return x
    
    def forward(self, data: Data, params: torch.Tensor):
        # x_input = data.x.clone() # for skip connection
        # Ensure data.x is float32
        data.x = data.x.float()
        encoded_x = self.graph_encoder(data)

        if self.is_autoencoder:
            latent_variables, decoded_reshaped_x = self.linear_autoencoder(encoded_x)
            estimated_latent_variables = self.mapping(params)
        else:
            latent_variables, estimated_latent_variables, decoded_reshaped_x = None, None, encoded_x

        estimated_x = self.graph_decoder(data, decoded_reshaped_x)

        # -- Global skip connection --
        # if self.is_skip_connection:
        #     if self.skip_proj is None:
        #         self.skip_proj = nn.Linear(x_input.shape[1], data.x.shape[1]).to(x_input.device)

        #     projected_skip = self.skip_proj(x_input)
        #     assert projected_skip.shape == data.x.shape
        #     data.x = data.x + 0.1 * projected_skip

        return estimated_x, latent_variables, estimated_latent_variables

    def reset_parameters(self):
        self.graph_encoder.reset_parameters()
        self.graph_decoder.reset_parameters()
        if self.linear_autoencoder is not None:
            self.linear_autoencoder.reset_parameters()

