#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader

from graphDatasetCreater import csv2graphDataset
import random
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

import os

# GraphConv
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(50, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 4)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hidden_channels=64)#.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

base_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data"
csv_path_list = {0:base_dir+'/SI/work.csv',1:base_dir+'/SI/meal_and_working_tools.csv',2:base_dir+'/SI/meal_while_working.csv',3:base_dir+'/SI/meal.csv'}
datasets,_ = csv2graphDataset(csv_path_list)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("dataset length : ", len(datasets))
# data = datasets[0]
# print("data 0 : ", data)
# G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])
# print(G.get_edge_data)
# pos = nx.spring_layout(G, k=0.3)
# nx.draw_networkx_edge_labels(G, pos)
# nx.draw_networkx(G, pos, with_labels=True, alpha=0.5)
# plt.show()
random.shuffle(datasets)
train_dataset = datasets[:int(len(datasets)*0.85)]
test_dataset = datasets[int(len(datasets)*0.85):]
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)



def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

for epoch in range(1, 10):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


# model_path = os.path.dirname(__file__) + '/model/SI_gcn-test.pt'
# torch.save(model.state_dict(),model_path)


