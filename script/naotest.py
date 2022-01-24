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





user_name = 'nao'
# train_data_type = 'row'
can_augment = True

ft_path = os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin'
graph_utils = graph_utilitys(fasttext_model=ft_path)

# 状態パターンごとのファイルを取得
user_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/"+user_name
files = glob.glob(user_dir + "/position_data/recognition*")
files.sort()
print(files)
pattern_num = len(files)
print('状態パターン数 : ', pattern_num)
csv_path_dict = {}
for i, file in enumerate(files):
    csv_path_dict[i] = file

if can_augment:
    augmented_csv_path_dict = augment_data_creater.augment(graph_utils, csv_path_dict)
    csv_path_dict_for_train = augmented_csv_path_dict
else:
    csv_path_dict_for_train = csv_path_dict

