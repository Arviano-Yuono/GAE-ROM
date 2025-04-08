import torch
from torch_geometric.data import Data
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons
from src.data.loader import GraphDataset

config = commons.get_config('configs/default.yaml')['model']['encoder']

class Encoder(torch.nn.Module):
    def __init__(self, config = config):
        super(Encoder, self).__init__()
        self.config = config
        #conv layers
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])
        #pool layer
        self.is_pooling = config['pool']['is_pooling']
        self.pool_method = config['pool']['type']
        self.in_channels = config['pool']['in_channels']
        self.batch_index = None
        self.unpool_info = None
        if self.is_pooling:
            if self.pool_method == 'EdgePool':
                from torch_geometric.nn import EdgePooling
                self.pooling_layer = EdgePooling(
                    in_channels=self.in_channels,
                    dropout=self.config['pool']['dropout']
                )
            elif self.pool_method == 'TopKPool':
                from torch_geometric.nn import TopKPooling
                self.pooling_layer = TopKPooling(
                    in_channels=self.in_channels,
                    ratio=self.config['pool']['ratio']
                )
            else:
                raise ValueError(f"Invalid pool method: {self.pool_method}")
        else:
            self.pool_method = 'None'
            self.pooling_layer = None
            
    def forward(self, data: Data):
        if data.x is None:
            raise ValueError("Input data.x is None. Please ensure data has node features.")

        # Apply all but the last convolution layer
        for conv, norm in zip(self.convolution_layers.convs[:-1], self.convolution_layers.batch_norms[:-1]):
            data.x = self.convolution_layers.act(conv(data.x, data.edge_index))
            data.x = norm(data.x)
            data.x = self.convolution_layers.dropout(data.x)

        if self.is_pooling:
            self.pool(data) 
            # Apply the last convolution layer
            pooled_x = self.convolution_layers.act(
                self.convolution_layers.convs[-1](self.pooled_x, self.pooled_edge_index))
            pooled_x = self.convolution_layers.batch_norms[-1](pooled_x)
            pooled_x = self.convolution_layers.dropout(pooled_x)
            return data, pooled_x, self.pooled_edge_index, self.pooled_edge_attr, self.unpool_info # the data returned is the data before pooling
        else:
            data.x = self.convolution_layers.act(
                self.convolution_layers.convs[-1](data.x, data.edge_index))
            data.x = self.convolution_layers.batch_norms[-1](data.x)
            data.x = self.convolution_layers.dropout(data.x)
            return data, data.x, data.edge_index, data.edge_attr, None # no unpooling info
            
    def pool(self, data: Data):
        if self.batch_index is None and data.batch is None:
            self.batch_index = torch.zeros(data.x.shape[0], dtype=torch.long)
        else:
            self.batch_index = data.batch

        if self.pool_method == 'EdgePool':
            self.pooled_x, self.pooled_edge_index, _, self.unpool_info = self.pooling_layer(data.x, data.edge_index, self.batch_index)
            self.pooled_edge_attr = torch.tensor(GraphDataset.compute_edge_distances_static(data.pos, self.pooled_edge_index))

        elif self.pool_method == 'TopKPool':
            self.pooled_x, self.pooled_edge_index, self.pooled_edge_attr, self.batch_index, self.perm, self.score = self.pooling_layer(x = data.x, 
                                                                                                                                       edge_index = data.edge_index, 
                                                                                                                                       edge_attr = data.edge_attr)
            self.unpool_info = self.perm

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

