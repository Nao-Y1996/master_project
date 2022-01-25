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
import augment_data_creater

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import random
import csv
import os
import glob
import fasttext
import sys

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
        loss = criterion(pred, batch.y)
        _, pred_labels = torch.max(pred, axis=1)
        correct_num += torch.sum(pred_labels == batch.y) # バッチの出力の中で正解ラベルと一致している物の個数を足していく→このエポックでの総正解数を得る
        total_data_len += len(pred_labels) # バッチサイズをiterator回足して学習データの総数を得る
        # loss.backward()
        # optimizer.step()
        total_loss += loss.item()
    # accuracy = float(correct_num)/total_data_len
    epoch_loss, epoch_accuracy = total_loss/total_data_len, float(correct_num)/total_data_len
    return epoch_loss, epoch_accuracy


# # userの選択
# user_name = input('enter user name \n')

# # 学習用データの選択
# train_data_type = input('enter train_data_type (ideal or row) \n')
# if (train_data_type=='ideal') or \
#    (train_data_type=='row') :
#     pass
# else:
#     sys.exit('enter train_data_type (ideal or row) \n')

# # データの拡張をするかどうかを選択
# can_augment = input('Do you augment data ? (y/n)\n')
# if can_augment=='y':
#     can_augment = True
# elif can_augment=='n':
#     can_augment = False
# else:
#     sys.exit('Do you augment data ? (y/n)\n')
users = ['yamada']#['comb-kusakari_ozawa', 'comb-kusakari_tou', 'comb-tou_ozawa']
train_data_type_list = ['ideal'] # ['ideal', 'row']
can_augment_list = ['y','n']
total_epoch = 50

for user_name in users:
    for train_data_type in train_data_type_list:
        for can_augment in can_augment_list:

            if can_augment=='y':
                can_augment = True
            elif can_augment=='n':
                can_augment = False
            else:
                sys.exit('Do you augment data ? (y/n)\n')

            print(f'~~~~~~~~~~~~~~~~~~~~~ {user_name} データタイプ={train_data_type} データ拡張={can_augment} ~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            ft_path = os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin'
            graph_utils = graph_utilitys(fasttext_model=ft_path)

            # 状態パターンごとのファイルを取得
            user_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/"+user_name
            files = glob.glob(user_dir + "/position_data/"+train_data_type+"_pattern_*")
            # print(files)
            files.sort()
            pattern_num = len(files)
            # print('状態パターン数 : ', pattern_num)
            csv_path_dict = {}
            for i, file in enumerate(files):
                csv_path_dict[i] = file

            if can_augment:
                augmented_csv_path_dict = augment_data_creater.augment(graph_utils, csv_path_dict, user_name=user_name[0])
                csv_path_dict_for_train = augmented_csv_path_dict
            else:
                csv_path_dict_for_train = csv_path_dict

            model_name = csv_path_dict_for_train[0].split('/')[-1].replace('pattern_0.csv','') # row_ / row_augmented_ / ideal_ / ideal_augmented_


            # グラフデータセットを作成
            datasets,_ = graph_utils.csv2graphDataset(csv_path_dict_for_train)
            print("dataset length : ", len(datasets))
            data = datasets[0]
            # print("data 0 : ", data)
            # G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])
            # print(G.get_edge_data)
            # pos = nx.spring_layout(G, k=0.3)
            # nx.draw_networkx_edge_labels(G, pos)
            # nx.draw_networkx(G, pos, with_labels=True, alpha=0.5)
            # plt.show()
            random.shuffle(datasets)
            train_dataset = datasets[:int(len(datasets)*0.5)]
            test_dataset = datasets[int(len(datasets)*0.5):]

            # バッチサイズの入力
            # batch_size = int(input('enter batch size \n'))
            # model_name += str(batch_size)+'_'
            # バッチ学習
            batch_size =4000# len(datasets)
            model_name += 'batch_'

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = NNConvNet(node_feature_dim=300, edge_feature_dim=3, output_dim=pattern_num).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters())


            print('-------train/test---------')
            train_loss_list = []
            train_acc_list = []
            test_loss_list = []
            test_acc_list = []
            for epoch in range(total_epoch):
                train_loss, train_acc = train(model, train_loader , optimizer, criterion)
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)

                test_loss, test_acc = test(model, test_loader)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                print(f'epoch = {epoch}')
                print(f'train loss = {train_loss}  train Accuracy = {train_acc}')
                print(f'test loss = {test_loss}  test Accuracy = {test_acc}')

            # モデルを保存
            model_path = user_dir + '/'+model_name+'nnconv.pt'
            torch.save(model.state_dict(),model_path)

            x = range(len(train_acc_list))
            # lossの描画
            fig = plt.figure()
            plt.plot(x, train_loss_list, color='b')
            plt.ylabel("Train Loss")
            # plt.show()
            fig.savefig(user_dir+"/"+model_name+"LossTrain.png")
            plt.close()

            fig = plt.figure()
            plt.plot(x, test_loss_list, color='y')
            plt.ylabel("Test Loss")
            # plt.show()
            fig.savefig(user_dir+"/"+model_name+"LossTest.png")
            plt.close()

            fig = plt.figure()
            plt.plot(x, train_loss_list, color='b', label='train')
            plt.plot(x, test_loss_list, color='y', label='test')
            plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=15)
            plt.ylabel("Loss")
            # plt.show()
            fig.savefig(user_dir+"/"+model_name+"Loss.png")
            plt.close()

            # accの描画
            fig = plt.figure()
            plt.plot(x, train_acc_list, color='b')
            plt.ylim(0.0, 1.0)
            plt.ylabel("Train Accuracy")
            # plt.show()
            fig.savefig(user_dir+"/"+model_name+"AccuracyTrain.png")
            plt.close()

            fig = plt.figure()
            plt.plot(x, test_acc_list, color='y')
            plt.ylim(0.0, 1.0)
            plt.ylabel("Test Accuracy")
            # plt.show()
            fig.savefig(user_dir+"/"+model_name+"AccuracyTest.png")
            plt.close()

            fig = plt.figure()
            plt.plot(x, train_acc_list, color='b', label='train')
            plt.plot(x, test_acc_list, color='y', label='test')
            plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=15)
            plt.ylim(0.0, 1.0)
            plt.ylabel("Accuracy")
            # plt.show()
            fig.savefig(user_dir+"/"+model_name+"Accuracy.png")
            plt.close()
            print()
            print()

            # print('--------test---------')
            # acc = test(model, test_loader)
            # print(f'Accuracy : {acc}')

