#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch_scatter import scatter_max
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from graph_converter import graph_utilitys

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import random
import csv
import os
import glob
import fasttext

class NNConvNet(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, output_dim):
        super(NNConvNet, self).__init__()
        self.edge_fc1 = nn.Linear(edge_feature_dim, node_feature_dim*32)
        self.nnconv1 = NNConv(node_feature_dim, 32, self.edge_fc1, aggr="mean")
        self.edge_fc2 = nn.Linear(edge_feature_dim, 32*48)
        self.nnconv2 = NNConv(32, 48, self.edge_fc2, aggr="mean")
        self.edge_fc3 = nn.Linear(edge_feature_dim, 48*64)
        self.nnconv3 = NNConv(48, 64, self.edge_fc3, aggr="mean")
        self.edge_fc4 = nn.Linear(edge_feature_dim, 64*128)
        self.nnconv4 = NNConv(64, 128, self.edge_fc4, aggr="mean")
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_dim)
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # print(x, edge_index, edge_attr, sep='\n')
        # print(f'0 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        # weight = self.nnconv1.nn(edge_attr).view(-1, self.nnconv1.in_channels_l, self.nnconv1.out_channels)
        # print(f'edge weight matrix shape = {np.shape(weight)}')
        x = self.nnconv1(x, edge_index, edge_attr)
        # print(f'1 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        x = F.relu(x)
        # print(f'2 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        # print(np.shape(self.nnconv2.nn(edge_attr)))
        x = self.nnconv2(x, edge_index, edge_attr)
        # print(f'3 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        x = F.relu(x)
        # print(f'4 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        # print(np.shape(self.nnconv3.nn(edge_attr)))
        x = self.nnconv3(x, edge_index, edge_attr)
        # print(f'5 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        x = F.relu(x)
        # print(f'6 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        # print(np.shape(self.nnconv4.nn(edge_attr)))
        x = self.nnconv4(x, edge_index, edge_attr)
        # print(f'7 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        x = F.relu(x)
        # print(f'8 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        x, _ = scatter_max(x, data.batch, dim=0)
        # print(f'9 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        x = self.fc1(x)
        # print(f'10 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        x = F.relu(x)
        # print(f'11 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        x = self.fc2(x)
        # print(f'12 ==> node:{np.shape(x)}, edge_index:{np.shape(edge_index)}, edge_attr:{np.shape(edge_attr)}')
        # print(asdf)
        return x

def train(model, iterator, optimizer, criterion):
    model.train()
    total_data_len = 0
    total_loss = 0
    correct_num = 0
    for batch in iterator:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        _, pred_labels = torch.max(pred, axis=1)
        # print(len(pred_labels), len(batch.y), sep='\n')
        correct_num += torch.sum(pred_labels == batch.y) # バッチの出力の中で正解ラベルと一致している物の個数を足していく→このエポックでの総正解数を得る
        total_data_len += len(pred_labels) # バッチサイズをiterator回足して学習データの総数を得る
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_loss, epoch_accuracy = total_loss/total_data_len, float(correct_num)/total_data_len
    return epoch_loss, epoch_accuracy

def test(model, iterator):
    model.eval()
    total_data_len = 0
    total_loss = 0
    correct_num = 0
    for batch in iterator:
        batch = batch.to(device)
        # optimizer.zero_grad()
        pred = model(batch)
        # loss = criterion(pred, batch.y)
        _, pred_labels = torch.max(pred, axis=1)
        correct_num += torch.sum(pred_labels == batch.y) # バッチの出力の中で正解ラベルと一致している物の個数を足していく→このエポックでの総正解数を得る
        total_data_len += len(pred_labels) # バッチサイズをiterator回足して学習データの総数を得る
        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item()
    accuracy = float(correct_num)/total_data_len
    return accuracy


ft_path = os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin'
graph_utils = graph_utilitys(fasttext_model=ft_path)

# 状態パターンごとのファイルを取得
base_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/2022-01-20/user_1/position_data"
file_list = []
files = glob.glob(base_dir + "/pattern*")
for file in files:
    if not 'augmented' in file:
        file_list.append(file)
file_list.sort()
csv_path_dict = {}
for i, file in enumerate(file_list):
    csv_path_dict[i] = file

# 学習データを拡張（augmented_pattern_n.csvを作成する）
remove_obj_names_list = [['sandwitch', 'soup', 'salada', 'book'], # pattern_0 : 仕事
                         ["laptop", "mouse", "keyboard", 'book'], # pattern_1 : 食事
                         ["laptop", "mouse", "keyboard", 'sandwitch', 'soup', 'salada']] # pattern_2 : 読書
remove_obj_ids_list = []
for remove_obj_names in remove_obj_names_list:
    remove_obj_ids = []
    for name in remove_obj_names:
        remove_obj_ids.append(graph_utils.OBJECT_NAME_2_ID[name])
    remove_obj_ids_list.append(remove_obj_ids)
data_num_list = []
for origin_csv, remove_obj_ids in zip(csv_path_dict.values(), remove_obj_ids_list):
    augmented_csv = origin_csv.replace('pattern', 'augmented_pattern')
    data_num = graph_utils.CreateAugmentedCSVdata(origin_csv, augmented_csv, remove_obj_ids)
    data_num_list.append(data_num)

# 拡張した学習データのデータ数を揃える（一番少ないものに合わせる。多いものはランダムに選択）
min_data_num = min(data_num_list)
print("データ数 : ",data_num_list, ' 最小 : ',min_data_num)
print(f'データ数を{min_data_num}に揃えます')
for k, v in csv_path_dict.items():
    csv_path_dict[k] = v.replace('pattern', 'augmented_pattern')
for file in csv_path_dict.values():
    with open(file) as f:
        csv_file = csv.reader(f)
        data = [[float(v) for v in row] for row in csv_file]
    data = random.sample(data, min_data_num)
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

# 拡張したデータを用いてグラフデータセットを作成
datasets,_ = graph_utils.csv2graphDataset(csv_path_dict)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("dataset length : ", len(datasets))
data = datasets[0]
print("data 0 : ", data)
# G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])
# print(G.get_edge_data)
# pos = nx.spring_layout(G, k=0.3)
# nx.draw_networkx_edge_labels(G, pos)
# nx.draw_networkx(G, pos, with_labels=True, alpha=0.5)
# plt.show()
random.shuffle(datasets)
train_dataset = datasets[:int(len(datasets)*0.85)]
test_dataset = datasets[int(len(datasets)*0.85):]
train_loader = DataLoader(train_dataset, batch_size=3000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=3000, shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NNConvNet(node_feature_dim=300, edge_feature_dim=3, output_dim=4
                  ).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


print('-------train---------')
train_loss_list = []
train_acc_list = []
for epoch in range(10):
    train_loss, train_acc = train(model, train_loader , optimizer, criterion)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    print(f'loss : {train_loss}  Accuracy : {train_acc}')

x, loss, acc = len(train_acc_list), train_loss_list, train_acc_list
# lossの描画
fig = plt.figure()
plt.plot(x, loss)
plt.ylabel("Loss")
fig.savefig("Loss.png")

# accの描画
fig = plt.figure()
plt.plot(x, loss)
plt.ylabel("Accuracy")
fig.savefig("Accuracy.png")

print('--------test---------')
acc = test(model, test_loader)
print(f'Accuracy : {acc}')


model_path = os.path.dirname(os.path.abspath(__file__)) + '/model/master_model_nnconv.pt'
torch.save(model.state_dict(),model_path)
