import torch
from torch_geometric.data import Data
from src.model.convolution_layers import ConvolutionLayers
from src.utils import commons
from src.data.loader import GraphDataset

config = commons.get_config('configs/default.yaml')['model']['encoder']

class GraphEncoder(torch.nn.Module):
    def __init__(self, config = config):
        super(GraphEncoder, self).__init__()
        self.config = config
        #conv layers
        self.convolution_layers = ConvolutionLayers(config['convolution_layers'])
        #pool layer
        self.is_pooling = config['pool']['is_pooling']
        self.pool_method = config['pool']['type']
        self.in_channels = config['pool']['in_channels']
        self.pooling_layer = []
        self.batch_index = None
        self.unpool_info = None
        if self.is_pooling:
            for ratio, in_channels in zip(self.config['pool']['ratio'], self.config['pool']['in_channels']):                
                if self.pool_method == 'EdgePool':
                    from torch_geometric.nn import EdgePooling

                    self.pooling_layer.append(EdgePooling(
                        in_channels=in_channels,
                        dropout=self.config['pool']['dropout']
                    ))

                elif self.pool_method == 'TopKPool':
                    from torch_geometric.nn import TopKPooling

                    self.pooling_layer.append(TopKPooling(
                        in_channels=in_channels,
                        ratio=ratio
                    ))

                elif self.pool_method == 'SAGPool':
                    from torch_geometric.nn import SAGPooling

                    if self.config['pool']['GNN_pool_type'] == 'GAT':
                        from torch_geometric.nn.conv import GATConv
                        self.pooling_layer.append(SAGPooling(
                            in_channels=in_channels,
                            ratio=ratio,
                            GNN_pool_type=GATConv(
                                in_channels=in_channels,
                                out_channels=in_channels,
                                heads=4,
                                dropout=self.config['pool']['dropout']
                            )
                        ))
                    else:
                        raise ValueError(f"Invalid GNN pool type: {self.config['pool']['GNN_pool_type']}")
                else:
                    raise ValueError(f"Invalid pool method: {self.pool_method}")
        else:
            for i in range(len(self.convolution_layers.convs)):
                self.pooling_layer.append(None)
            self.pool_method = 'None'
            
    def forward(self, data: Data):
        if data.x is None:
            raise ValueError("Input data.x is None. Please ensure data has node features.")
        for conv, norm, pooling_layer in zip(self.convolution_layers.convs, self.convolution_layers.batch_norms, self.pooling_layer):
            # check if conv take edge_attr as input
            if hasattr(conv, 'edge_conv'):
                data.x = self.convolution_layers.act(conv(x = data.x, edge_index = data.edge_index, edge_attr = data.edge_attr))
            else:
                data.x = self.convolution_layers.act(conv(x = data.x, edge_index = data.edge_index))
            data.x = norm(data.x)
            data.x = self.convolution_layers.dropout(data.x)
            if self.is_pooling:
                self.pool(data, pooling_layer) 

        return data, data.x, data.edge_index, data.edge_attr, self.unpool_info # no unpooling info
            
    def pool(self, data: Data, pooling_layer):
        if self.batch_index is None or data.batch is None:
            self.batch_index = torch.zeros(data.x.shape[0], dtype=torch.long)
        else:
            self.batch_index = data.batch

        if self.pool_method == 'EdgePool':
            data.x, data.edge_index, _, unpool_info = pooling_layer(data.x, data.edge_index, self.batch_index)
            data.edge_attr = torch.tensor(GraphDataset.compute_edge_distances_static(data.pos, data.edge_index))
            self.unpool_info.append(unpool_info)

        elif self.pool_method == 'TopKPool':
            data.x, data.edge_index, data.edge_attr, self.batch_index, self.perm, self.score = pooling_layer(x = data.x, edge_index = data.edge_index, edge_attr = data.edge_attr)
            self.unpool_info.append(self.perm)

        elif self.pool_method == 'SAGPool':
            data.x, data.edge_index, data.edge_attr, self.batch_index, _ = pooling_layer(x = data.x, edge_index = data.edge_index, edge_attr = data.edge_attr)
            self.unpool_info.append(None)

    def reset_parameters(self):
        self.convolution_layers.reset_parameters()

