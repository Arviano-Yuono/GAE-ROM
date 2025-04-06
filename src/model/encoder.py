import torch
from torch_geometric.nn import EdgePooling
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
        self.input_dim = config['dim']
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])
        #pool layer
        self.is_pooling = config['pool']['is_pooling']
        self.pool_method = config['pool']['type']
        self.in_channels = config['pool']['in_channels']
        self.unpool_info = None

        if self.is_pooling:
            if self.pool_method == 'EdgePool':
                self.pool = EdgePooling(
                    in_channels=self.in_channels,
                    dropout=self.config['pool']['dropout']
                )
            else:
                raise ValueError(f"Invalid pool method: {self.pool_method}")
            
    def forward(self, data: Data):
        if data.x is None:
            raise ValueError("Input data.x is None. Please ensure data has node features.")

        # Apply all but the last convolution layer
        for conv in self.convolution_layers.convs[:-1]:
            data.x = self.convolution_layers.act(conv(data.x, data.edge_index, data.edge_attr))
            data.x = self.convolution_layers.dropout(data.x)
            

        if self.is_pooling:
            if data.batch is None:
                batch_vector = torch.zeros(data.x.shape[0], dtype=torch.long)
            else:
                batch_vector = data.batch
                
            # Apply pooling and get the new edge indices
            pooled_x, pooled_edge_index, _, self.unpool_info = self.pool(
                x=data.x,
                edge_index=data.edge_index,
                batch=batch_vector
            )
            
            # Update edge attributes to match the new edge indices
            # if hasattr(self.unpool_info, 'edge_index') and data.edge_attr is not None:
            #     kept_edge_indices = self.unpool_info.edge_index[1]
            pooled_edge_attr = torch.tensor(GraphDataset.compute_edge_distances_static(data.pos, pooled_edge_index))

        
            # Apply the last convolution layer
            data.x = self.convolution_layers.act(
                self.convolution_layers.convs[-1](data.x, pooled_edge_index, pooled_edge_attr))

            # flattened data of data.x
            flattened_data = torch.flatten(data.x) # shape: (num_nodes * pool_ratio * hidden_channels)
            self.flattened_dim = flattened_data.shape
            return data, flattened_data, self.unpool_info, pooled_edge_attr
    
    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

