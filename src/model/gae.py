from model.encoder import Encoder
from model.decoder import Decoder
import torch.nn as nn
from torch_geometric.data import Data

class GAE(nn.Module):
    def __init__(self, config):
        super(GAE, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, data: Data):
        data = self.encoder(data)
        data = self.decoder(data)
        return data

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()


