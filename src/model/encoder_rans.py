import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from src.model.convolution_layers import ConvolutionLayers
import torch.nn as nn
from src.utils import commons

class GraphEncoder(torch.nn.Module):
    def __init__(self, config, device = None):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.device = device

        #conv layers
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])

        #classifier
        self.classifier_config = config['classifier']
        self.latent_dim_size = self.classifier_config['layer_vec'][-1]
        act_name = self.classifier_config['act']
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
        
        if self.classifier_config['is_classifier']:
            self.classifier = nn.Sequential()
            self.classifier.add_module('classifier_layer_0', nn.Linear(config['convolution_layers']['hidden_channels'][-1], self.classifier_config['layer_vec'][0], device=self.device))
            self.classifier.add_module('classifier_layer_0_act', act)
            for k in range(len(self.classifier_config['layer_vec'])-2):
                self.classifier.add_module(f'classifier_layer_{k+1}', nn.Linear(self.classifier_config['layer_vec'][k], self.classifier_config['layer_vec'][k+1], device=self.device))
                self.classifier.add_module(f'classifier_layer_{k+1}_act', act)
            self.classifier.add_module('classifier_layer_latent_dim', nn.Linear(self.classifier_config['layer_vec'][-2], self.latent_dim_size, device=self.device))
            self.classifier = self.classifier.to(self.device)

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
        # print(f"x shape original: {x.shape}")
        idx = 0
        for i, (conv, norm) in enumerate(zip(self.convolution_layers.convs, self.convolution_layers.batch_norms)):
            # check if conv take edge_attr as input
            if self.convolution_layers.config['type'] in ['ChebConv', 'GCNConv']:
                x = conv(x, data.edge_index, data.edge_weight)
            elif self.convolution_layers.config['type'] in ['GATConv', 'GMMConv']:
                x = conv(x, data.edge_index, data.edge_attr)
            elif self.convolution_layers.config['type'] in ['SAGE']:
                x = conv(x, data.edge_index)
            if self.convolution_layers.is_skip_connection:
                x = x + data.x
            if i != len(self.convolution_layers.convs) - 2:
                x = self.convolution_layers.act(x)
            if norm is not None:
                x = norm(x)
            x = self.convolution_layers.dropout(x)
            # print(f"x shape after conv: {x.shape}")
            idx += 1
        # print(f"x shape after conv: {x.shape}")
        if self.classifier_config['is_classifier']:
            pooled_x = global_mean_pool(x, data.batch)
            # print(f"pooled_x shape: {pooled_x.shape}")
            return self.classifier(pooled_x)
        else:
            return x
        
    def reset_parameters(self):
        self.convolution_layers.reset_parameters()


