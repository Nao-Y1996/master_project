#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import csv
import os
from sklearn.preprocessing import minmax_scale
import fasttext
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
class DictConstrustionError(Exception):
    pass


class graph_utilitys():
    def __init__(self, fasttext_model):
        self.ID_2_OBJECT_NAME = {
                    0:"face", 1:"bottle", 2:"wine glass", 3:"cup", 4:"fork", 5:"knife", 6:"spoon", 7:"bowl",
                    8:"banana", 9:"apple", 10:"sandwich", 11:"orange", 12:"broccoli", 13:"carrot", 14:"hot dog", 15:"pizza", 16:"donut",
                    17:"cake", 18:"chair", 19:"sofa", 20:"pottedplant", 21:"bed", 22:"diningtable", 23:"toilet", 24:"tvmonitor", 25:"laptop",
                    26:"mouse", 27:"remote", 28:"keyboard", 29:"cell phone", 30:"microwave", 31:"oven", 32:"toaster", 33:"sink", 34:"refrigerator",
                    35:"book", 36:"clock", 37:"vase", 38:"scissors", 39:"teddy bear", 40:"hair drier", 41:"toothbrush"
                    }
        self.ft = fasttext.load_model(fasttext_model)

    def changeID_2_OBJECT_NAME(self, obj_name_changer_dict):
        """
        if change pattern is bellow
        'sandwich'-->'toast', 'robot'-->'camera'
        input would be
        obj_name_changer_dict = {'sandwich':'toast','robot':'camera'}
        """
        for origin_obj_name in obj_name_changer_dict.keys():
            # 変更したい物体名がID_2_OBJECT_NAMEのvalueに存在するかチェック
            if origin_obj_name in self.ID_2_OBJECT_NAME.values():
                # 変更したい名前のkey(id)をID_2_OBJECT_NAMEから見つける
                ids = [k for k, v in self.ID_2_OBJECT_NAME.items() if v == origin_obj_name]
                if len(ids) == 1:
                    # 物体名を変更する
                    id = ids[0]
                    self.ID_2_OBJECT_NAME[id] = obj_name_changer_dict[origin_obj_name]
                else:
                    raise DictConstrustionError('変更したい物体名のidが、ID_2_OBJECT_NAME内に複数見つかりました')
            else:
                raise DictConstrustionError('変更したい物体名が、ID_2_OBJECT_NAME内に見つかりません。')

    def convertData2graph(self, position_data, label, include_names=False):
        obj_num = int(len(position_data)/4)
        #物体の数が0 or 1の時はグラフ作成できない
        if (obj_num==0) or (obj_num==1):
            return None, None
        names = []
        nodes_features = []
        positions = []
        for obj in np.array(position_data).reshape(-1,4):
            obj_id = obj[0]
            name = self.ID_2_OBJECT_NAME[obj_id]
            try:
                word_v = self.ft.get_word_vector(name).tolist()
            except KeyError:
                continue
            if include_names:
                names.append(name)
            nodes_features.append(word_v)
            x, y, z = obj[1], obj[2], obj[3]
            positions.append([x, y, z])
        x = torch.tensor(nodes_features,  dtype=torch.float)
        # print('nodes shape : \n',x.shape)
        # print('positions : \n', np.array(positions))

        # calculate distanse obj2obj
        position_dist_matrix = [[0 for i in range(obj_num)] for j in range(obj_num)]
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                dist =((np.linalg.norm(np.array(pos1) - np.array(pos2))).tolist())
                position_dist_matrix[i][j] = dist
        position_normarized_dist_matrix = np.reshape(minmax_scale(np.array(position_dist_matrix).flatten()), (obj_num,obj_num))
        # print('position_dist_matrix : \n',np.array(position_dist_matrix))
        # print('position_normarized_dist_matrix : \n',position_normarized_dist_matrix)

        # create edge_index, edge_feature
        edges = []
        edge_features = []
        for i, _ in enumerate(position_dist_matrix):
            for j, _ in enumerate(position_dist_matrix):
                dist = position_dist_matrix[i][j]
                normarized_dist = position_normarized_dist_matrix[i][j]
                # ここのif文にedgeを作る条件を入れる
                if i!=j: # 自己ループはなし
                    if i==0 or j==0:# 基準(camera, face等)と物体は必ずつなぐ
                        edges.append([i,j])
                        edge_features.append([normarized_dist])
                        continue
                    if dist <=0.3:
                        edges.append([i,j])
                        edge_features.append([normarized_dist])

        edge_index = torch.tensor(np.array(edges).T.tolist(), dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        graph = Data(x=x, y=y, edge_index=edge_index,edge_attr=edge_attr)
        return graph, names
    
    def csv2graphDataset(self, csv_files, include_names=False):
        """
        csv_files = {0:'1-1.csv', 1:'3-2.csv',
                        2:'5-3.csv', 3:'7-4.csv'}
        """
        obj_names_sets = []
        datasets = []
        for num in range(len(csv_files)):
            file_path = csv_files[num]
            with open(file_path) as f:
                csv_file = csv.reader(f)
                positions_data = [[float(v) for v in row] for row in csv_file]
            for row, position_data in enumerate(positions_data):
                graph, obj_names = self.convertData2graph(position_data, label=num, include_names=include_names)
                if graph is not None:
                    datasets.append(graph)
                    if len(obj_names)!=0:
                        obj_names_sets.append(obj_names)
        return datasets, obj_names_sets
    
    def visualize_graph(self, graph, node_labels, save_graph_name=None, show_graph=True):
        # plt.close()
        G = to_networkx(graph, node_attrs=['x'], edge_attrs=['edge_attr'])
        mapping = {k: v for k, v in zip(G.nodes, node_labels)}
        G = nx.relabel_nodes(G, mapping)
        c_list = ['skyblue' if n=='face' else 'orange' for n in G.nodes()]
        nx.draw_spring(G, with_labels=True, width = 3, edge_color="gray", node_color=c_list, node_size=2000)
        if save_graph_name is not None:
            plt.savefig(save_graph_name)
        if show_graph:
            # plt.show()
            plt.pause(0.1)
            plt.clf()