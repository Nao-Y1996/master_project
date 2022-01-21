#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch_scatter import scatter_max
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, NNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear

import os


# GCNConv
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(300, 200)
        self.conv2 = GCNConv(200, 100)
        self.conv3 = GCNConv(100, 30)
        self.lin = Linear(30, 4)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

class classificator():
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.dirname(__file__) + '/model/'+ model # 'SI_gcn.pt'
        self.loading_model =  GCN().to(self.device)
        self.loading_model.load_state_dict(torch.load(model_path))

    def classificate(self, graph):
        input_graph = [graph]
        input_loader = DataLoader(input_graph, batch_size=1, shuffle=True)

        self.loading_model.eval()
        sm = nn.Softmax(dim=1)
        for data in input_loader:
            data = data.to(self.device)
            pred = self.loading_model(data.x, data.edge_index, data.batch)
            probability = sm(pred).tolist()[0]
            # print(probability)
            return probability