import torch
import gc
from torch_geometric.data import Data
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons
import torch.nn as nn

config = commons.get_config('configs/default.yaml')['model']['decoder']

class GraphDecoder(torch.nn.Module):
    def __init__(self, config = config, device=None):
        super(GraphDecoder, self).__init__()
        self.config = config
        #conv layers
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])

        # Linear autoencoder components
        self.linear_decoder = None
        self.autoencoder_config = None
        self.is_autoencoder = False
        
        self.device = device
        self.autoencoder_config = config['autoencoder_decoder']
        self.is_autoencoder = self.autoencoder_config['is_autoencoder']

    def _initialize_linear_decoder(self, target_shape, device=None):
        """Initialize linear decoder layers if not already done."""
        if self.linear_decoder is not None:
            return
            
        # Calculate output dimension: num_nodes * hidden_channels[-1]
        output_dim = target_shape[0] * target_shape[1]
        
        # Get activation function
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
        
        # Build decoder layers
        self.linear_decoder = nn.Sequential()
        self.linear_decoder.add_module('decoder_layer_latent_dim', 
                                      nn.Linear(self.autoencoder_config['latent_dim'], 
                                               self.autoencoder_config['decoder_layers'][0], device=self.device))
        self.linear_decoder.add_module('decoder_layer_latent_dim_act', act)
        
        for i in range(len(self.autoencoder_config['decoder_layers'])-1):
            self.linear_decoder.add_module(f'decoder_layer_{i}', 
                                          nn.Linear(self.autoencoder_config['decoder_layers'][i], 
                                                   self.autoencoder_config['decoder_layers'][i+1], device=self.device))
            self.linear_decoder.add_module(f'decoder_layer_{i}_act', act)
        
        self.linear_decoder.add_module('decoder_layer_output', 
                                      nn.Linear(self.autoencoder_config['decoder_layers'][-1], output_dim, device=self.device))
        self.linear_decoder.add_module('decoder_layer_output_act', act)
        self.linear_decoder = self.linear_decoder.to(self.device)

    def forward(self, data: Data, latent_variables: torch.Tensor):
        """
        Forward pass through linear decoder and graph decoder.
        
        Args:
            data: Input graph data (for edge information)
            latent_variables: Latent variables to decode
            
        Returns:
            Decoded/reconstructed features
        """
        # If autoencoder is enabled, decode from latent variables
        if self.is_autoencoder:
            # We need to determine the expected shape for the graph decoder
            # This should match the output shape of the graph encoder
            # For now, we'll use a placeholder shape that will be updated
            # You might need to pass this information or store it during encoding
            expected_shape = (data.x.shape[0], self.convolution_layers.config['hidden_channels'][-1])
            
            self._initialize_linear_decoder(expected_shape, device=latent_variables.device)
            
            # Decode from latent space
            decoded_flat = self.linear_decoder(latent_variables)
            decoded_reshaped_x = decoded_flat.reshape(expected_shape)
        else:
            # If no autoencoder, use the input directly
            decoded_reshaped_x = latent_variables
        
        # Graph decoding
        x = decoded_reshaped_x
        idx = 0
        for conv, norm in zip(self.convolution_layers.convs, self.convolution_layers.batch_norms):
            if self.convolution_layers.config['type'] in ['Cheb', 'GCN']:
                x = conv(x, data.edge_index, data.edge_weight)
            elif self.convolution_layers.config['type'] in ['GAT', 'GMM']:
                x = conv(x, data.edge_index, data.edge_attr)

            if (idx != len(self.convolution_layers.convs) - 2):
                x = self.convolution_layers.act(x)

            if self.convolution_layers.is_skip_connection:
                x = x + decoded_reshaped_x

            if norm is not None:
                x = norm(x)
            x = self.convolution_layers.dropout(x)
            idx += 1
        return x

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()
        if self.linear_decoder is not None:
            for module in self.linear_decoder.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

