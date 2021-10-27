#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
# import torch.nn.functional as F
 
# from torch_geometric.nn import GraphConv, GCNConv, NNConv
# from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
# from torch_geometric.datasets import KarateClub
# import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
# import random
# import csv
import os
from sklearn.preprocessing import minmax_scale


import fasttext
import fasttext.util
# fasttext.util.download_model('en', if_exists='ignore')  # English
# ft = fasttext.load_model('cc.en.300.bin')
# fasttext.util.reduce_model(ft, 50)
# ft.save_model('cc.en.50.bin')
# model_path = '../../Word/crawl-300d-2M-subword/crawl-300d-2M-subword.bin'
model_path = os.path.dirname(__file__) +'/w2v_model/cc.en.50.bin'
ft = fasttext.load_model(model_path)

# OBJECT_ID_LIST = {0:"bottle", 1:"wine glass", 2:"cup", 3:"fork", 4:"knife", 5:"spoon", 6:"bowl", 7:"banana", 8:"apple", 9:"sandwich", 10:"orange",
#                        11:"broccoli", 12:"carrot", 13:"hot dog", 14:"pizza", 15:"donut", 16:"cake", 17:"chair", 18:"sofa", 19:"pottedplant", 20:"bed",
#                        21:"diningtable", 22:"toilet", 23:"tvmonitor", 24:"laptop", 25:"mouse", 26:"remote", 27:"keyboard", 28:"cell phone", 29:"microwave", 30:"oven",
#                        31:"toaster", 32:"sink", 33:"refrigerator", 34:"book", 35:"clock", 36:"vase", 37:"scissors", 38:"teddy bear", 39:"hair drier", 40:"toothbrush"}
# OBJECT_ID_LIST = {"face":0, "bottle":1, "wine glass":2, "cup":3, "fork":4, "knife":5,     "spoon":6, "bowl":7, "banana":8, "apple":9, "sandwich":10, "orange":11,"broccoli":12, "carrot":13, "hot dog":14, "pizza":15, "donut":16, "cake":17,"chair":18, "sofa":119, "pottedplant":20, "bed":21, "diningtable":22, "toilet":23,"tvmonitor":24, "laptop":25, "mouse":26, "remote":27, "keyboard":28, "cell phone":29, "microwave":30, "oven":31, "toaster":32, "sink":33, "refrigerator":34, "book":35, "clock":36, "vase":37, "scissors":38, "teddy bear":39, "hair drier":40, "toothbrush":41}

ID_2_OBJECT_NAME = {  0:"robot", 1:"bottle", 2:"wine glass", 3:"cup", 4:"fork", 5:"knife", 6:"spoon", 7:"bowl",
                    8:"banana", 9:"apple", 10:"sandwich", 11:"orange", 12:"broccoli", 13:"carrot", 14:"hot dog", 15:"pizza", 16:"donut",
                    17:"cake", 18:"chair", 19:"sofa", 20:"pottedplant", 21:"bed", 22:"diningtable", 23:"toilet", 24:"tvmonitor", 25:"laptop",
                    26:"mouse", 27:"remote", 28:"keyboard", 29:"cell phone", 30:"microwave", 31:"oven", 32:"toaster", 33:"sink", 34:"refrigerator",
                    35:"book", 36:"clock", 37:"vase", 38:"scissors", 39:"teddy bear", 40:"hair drier", 41:"toothbrush"
                    }

def convertData2graph(position_data, label, include_names=False):
    obj_num = int(len(position_data)/4)
    #物体の数が0 or 1の時はグラフ作成できない
    if (obj_num==0) or (obj_num==1):
        return None
    # 先頭のデータが0.0でない（基準のデータが含まれていない）時はスキップ
    # if position_data[0]!=0.0:
    #     return None
    # create nodes (obj*num * 1)
    # create positions data (obj_num * 3)
    names = []
    nodes_features = []
    positions = []
    for obj in np.array(position_data).reshape(-1,4):
        obj_id = obj[0]
        name = ID_2_OBJECT_NAME[obj_id]
        try:
            word_v = ft.get_word_vector(name).tolist()
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

    # calculate distanse obj-obj
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
    # print('edge_index : \n', np.array(edge_index))
    # print('edge_attr : \n',edge_attr)
    graph = Data(x=x, y=y, edge_index=edge_index,edge_attr=edge_attr)
    return graph, names

def visualize_graph(graph, node_labels, save_graph_name=None, show_graph=True):
    G = to_networkx(graph, node_attrs=['x'], edge_attrs=['edge_attr'])
    mapping = {k: v for k, v in zip(G.nodes, node_labels)}
    G = nx.relabel_nodes(G, mapping)
    c_list = ['skyblue' if n=='robot' else 'orange' for n in G.nodes()]
    nx.draw_spring(G, with_labels=True, width = 7, edge_color="gray", node_color=c_list, node_size=2000)
    if save_graph_name is not None:
        plt.savefig(save_graph_name)
    if show_graph:
        # plt.show()
        plt.pause(0.1)
        plt.clf()