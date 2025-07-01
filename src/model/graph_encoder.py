import torch
from torch_geometric.data import Data
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons
import torch.nn as nn

config = commons.get_config('configs/default.yaml')['model']['encoder']

class GraphEncoder(torch.nn.Module):
    def __init__(self, config = config, device = None):
        super(GraphEncoder, self).__init__()
        self.config = config
        #conv layers
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])

        # Linear autoencoder components
        self.linear_encoder = None
        self.autoencoder_config = None
        self.is_autoencoder = False

        self.device = device
        self.autoencoder_config = config['autoencoder_encoder']
        self.is_autoencoder = self.autoencoder_config['is_autoencoder']

    def _initialize_linear_encoder(self, x):
        """Initialize linear encoder layers if not already done."""
        if self.linear_encoder is not None:
            return
            
        input_dim = x.shape[0] * x.shape[1]
        
        act_name = self.autoencoder_config['act']
        if act_name.lower() == 'relu':
            act = nn.ReLU()
        elif act_name.lower() == 'tanh':
            act = nn.Tanh()
        elif act_name.lower() == 'sigmoid':
            act = nn.Sigmoid()
        elif act_name.lower() == 'leaky_relu':
            act = nn.LeakyReLU()
        elif act_name.lower() == 'elu':
            act = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {act_name}")
        
        self.linear_encoder = nn.Sequential()
        self.linear_encoder.add_module('encoder_layer_0', 
                                      nn.Linear(input_dim, self.autoencoder_config['encoder_layers'][0], device=self.device))
        self.linear_encoder.add_module('encoder_layer_0_act', act)
        
        for i in range(len(self.autoencoder_config['encoder_layers'])-1):
            self.linear_encoder.add_module(f'encoder_layer_{i+1}', 
                                          nn.Linear(self.autoencoder_config['encoder_layers'][i], 
                                                   self.autoencoder_config['encoder_layers'][i+1], device=self.device))
            self.linear_encoder.add_module(f'encoder_layer_{i+1}_act', act)
        
        self.linear_encoder.add_module('encoder_layer_latent_dim', 
                                      nn.Linear(self.autoencoder_config['encoder_layers'][-1], 
                                               self.autoencoder_config['latent_dim'], device=self.device))
        
        # Move to the same device as input
        self.linear_encoder = self.linear_encoder.to(self.device)

    def forward(self, data: Data):
        """
        Forward pass through graph encoder and optionally linear encoder.
        
        Args:
            data: Input graph data
            
        Returns:
            If autoencoder is enabled: latent variables
            If autoencoder is disabled: graph-encoded features
        """
        x = data.x
        idx = 0
        for conv, norm in zip(self.convolution_layers.convs, self.convolution_layers.batch_norms):
            # check if conv take edge_attr as input
            if self.convolution_layers.config['type'] in ['Cheb', 'GCN']:
                x = self.convolution_layers.act(conv(x, data.edge_index, data.edge_weight))
            elif self.convolution_layers.config['type'] in ['GAT', 'GMM']:
                print(f"data.edge_attr shape: {data.edge_attr.shape}")
                print(f"x shape: {x.shape}")
                print(f"data.edge_index shape: {data.edge_index.shape}")
                x = self.convolution_layers.act(conv(x, data.edge_index, data.edge_attr))
            if self.convolution_layers.is_skip_connection:
                x = x + data.x
            if norm is not None:
                x = norm(x)
            x = self.convolution_layers.dropout(x)
            idx += 1
        
        # If autoencoder is enabled, encode to latent variables
        if self.is_autoencoder:
            self._initialize_linear_encoder(x)
            
            flattened_x = x.view(-1)   # global mean pooling
            
            # Ensure the flattened tensor has the correct shape for the linear layer
            if flattened_x.dim() == 1:
                flattened_x = flattened_x.unsqueeze(0)  # Add batch dimension if needed
            
            # Encode to latent space
            latent_variables = self.linear_encoder(flattened_x)
            return latent_variables
        else:
            # Return graph-encoded features
            return x
        
    def reset_parameters(self):
        self.convolution_layers.reset_parameters()
        if self.linear_encoder is not None:
            for module in self.linear_encoder.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

