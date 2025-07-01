import gc
from src.model.encoder_rans import GraphEncoder
from src.model.decoder_rans import GraphDecoder
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from src.utils.commons import get_activation_function

class GAE(nn.Module):
    def __init__(self, config, num_graphs = None, device = None):
        super(GAE, self).__init__()
        self.save_config = config
        self.config = self.save_config['model']

        self.graph_encoder = GraphEncoder(self.config['encoder'], device=device)
        self.graph_decoder = GraphDecoder(self.config['decoder'], device=device)

        self.maptovec = nn.ModuleList()
        self.act_map = get_activation_function(self.config['maptovec']['act'])
        # Ensure activation function is valid
        if self.act_map is None:
            raise ValueError(f"Invalid activation function: {self.config['maptovec']['act']}")
        for k in range(len(self.config['maptovec']['layer_vec'])-1):
            self.maptovec.append(nn.Linear(self.config['maptovec']['layer_vec'][k], self.config['maptovec']['layer_vec'][k+1], device=device))
        self.maptovec = self.maptovec.to(device)
        
        self.skip_proj = None # for skip connection
        # self.is_skip_connection = self.config['autoencoder']['is_skip_connection']

    def encode(self, data: Data):
        """
        Encode the input data through the graph encoder and optionally through the linear encoder.
        
        Args:
            data: Input graph data
            
        Returns:
            If autoencoder is enabled: latent variables
            If autoencoder is disabled: graph-encoded features
        """
        if data.x is not None:
            data.x = data.x.float()
        
        # The encoder now handles both graph encoding and linear encoding
        encoded_output = self.graph_encoder(data)
        return encoded_output

    def mapping(self, params):
        """
        Map input parameters to latent space using the mapping network.
        
        Args:
            params: Input parameters tensor
            
        Returns:
            tensor: Estimated latent variables
        """
        # Ensure input is float32
        params = params.float()
        
        idx = 0
        for layer in self.maptovec:
            if (idx == len(self.maptovec) - 1): 
                # Last layer doesn't need activation function because it's the latent var
                params = layer(params)
            else: 
                params = self.act_map(layer(params))
            idx += 1
        return params

    def decode(self, data: Data, encoded_output):
        """
        Decode the encoded output through the graph decoder.
        
        Args:
            data: Input graph data (for edge information)
            encoded_output: Encoded output from encoder (latent variables or graph features)
            
        Returns:
            tensor: Decoded/reconstructed features
        """
        # The decoder now handles both linear decoding and graph decoding
        estimated_x = self.graph_decoder(data, encoded_output)
        return estimated_x

    def forward(self, data: Data):
        """
        Forward pass through the complete GAE model.
        
        Args:
            data: Input graph data
            
        Returns:
            tuple: (estimated_x, latent_variables, estimated_latent_variables)
        """
        # Encode the data
        encoded_output = self.encode(data)
        
        # Map parameters to latent space
        if hasattr(data, 'params') and data.params is not None:
            estimated_latent_variables = self.mapping(data.params)
        else:
            estimated_latent_variables = None
        
        # Decode the features
        estimated_x = self.decode(data, encoded_output)

        # -- Global skip connection (commented out) --
        # if self.is_skip_connection:
        #     if self.skip_proj is None:
        #         self.skip_proj = nn.Linear(x_input.shape[1], data.x.shape[1]).to(x_input.device)
        #     projected_skip = self.skip_proj(x_input)
        #     assert projected_skip.shape == data.x.shape
        #     data.x = data.x + 0.1 * projected_skip

        return estimated_x, encoded_output, estimated_latent_variables

    def reset_parameters(self):
        """Reset all parameters in the model."""
        self.graph_encoder.reset_parameters()
        self.graph_decoder.reset_parameters()

