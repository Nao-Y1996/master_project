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
import sys


def augment(graph_utils, origin_csv_path_dict, user_name):
    augmented_csv_path_dict = {}
    # 学習データを拡張  (train_data_type)_augmented_pattern.csvを作成する
    if user_name=='k':

        # kusakari
        remove_obj_names_list = [['book', 'mouse', 'keyboard', 'tvmonitor'], # pattern_0 : 仕事
                                ["laptop", "tvmonitor", 'book'], # pattern_1 : 昼食
                                ["soup"]] # pattern_2 : 読書
    elif user_name=='o':
        # ozawa
        remove_obj_names_list = [['tvmonitor', 'book'], # pattern_0 : 仕事
                                ['soup', 'book'], # pattern_1 : 昼食
                                ['laptop', 'soup']] # pattern_2 : 読書
    elif user_name=='t':
        # tou
        remove_obj_names_list = [['book'], # pattern_0 : 仕事
                                ['tvmonitor', 'laptop'], # pattern_1 : 昼食
                                ['keyboard', 'mouse']] # pattern_2 : 読書
    elif user_name=='y':
        # yamada
        remove_obj_names_list = [['book', 'mouse', 'keyboard', 'sandwich', 'soup', 'salada'], # pattern_0 : 仕事
                                ['book', 'mouse', 'keyboard', 'tvmonitor', 'laptop', 'soup', 'salada'], # pattern_1 : 昼食
                                ['mouse', 'keyboard', 'tvmonitor', 'laptop', 'sandwich', 'soup', 'salada']] # pattern_2 : 読書
    else:
        sys.exit('select collect user')
        # ['book', 'mouse', 'keyboard', 'tvmonitor', 'laptop', 'sandwich', 'soup', 'salada']


    remove_obj_ids_list = []
    for remove_obj_names in remove_obj_names_list:
        remove_obj_ids = []
        for name in remove_obj_names:
            remove_obj_ids.append(graph_utils.OBJECT_NAME_2_ID[name])
        remove_obj_ids_list.append(remove_obj_ids)
    data_num_list = []
    for origin_csv, remove_obj_ids in zip(origin_csv_path_dict.values(), remove_obj_ids_list):
        augmented_csv = origin_csv.replace('pattern', 'augmented_pattern') # (train_data_type)_pattern.csv --> (train_data_type)_augmented_pattern.csv
        # データを拡張して保存　＆　データ数を返す
        data_num = graph_utils.CreateAugmentedCSVdata(origin_csv, augmented_csv, remove_obj_ids)
        data_num_list.append(data_num)

    # 拡張した学習データのデータ数を揃える（一番少ないものに合わせる。多いものはランダムに選択）
    min_data_num = min(data_num_list)
    print("データ数 : ",data_num_list, ' 最小 : ',min_data_num)
    print(f'データ数を{min_data_num}に揃えます')
    for k, v in origin_csv_path_dict.items():
        augmented_csv_path_dict[k] = v.replace('pattern', 'augmented_pattern')
    
    # データ数が揃えられた拡張データを保存し直す
    for file in augmented_csv_path_dict.values():
        with open(file) as f:
            csv_file = csv.reader(f)
            data = [[float(v) for v in row] for row in csv_file]
        data = random.sample(data, min_data_num)
        with open(file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    
    return augmented_csv_path_dict