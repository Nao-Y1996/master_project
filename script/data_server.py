#!/usr/bin/python3
# -*- coding: utf-8 -*-

import socket
import datetime
import pickle
from graph_converter import graph_utilitys
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

from classificator_gcn import classificator
print("start-----------------------------------")
cf = classificator(model='SI_gcn-w300-30cm.pt')
graph_utils = graph_utilitys(fasttext_model='cc.en.300.bin')

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
IP_ADDRESS = s.getsockname()[0]

# AF = IPv4 という意味
# TCP/IP の場合は、SOCK_STREAM を使う
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # IPアドレスとポートを指定
    s.bind((IP_ADDRESS, 12345))
    # s.bind(('127.0.0.1', 50010))
    # 1 接続
    s.listen(1)
    # connection するまで待つ
    while True:
        # 誰かがアクセスしてきたら、コネクションとアドレスを入れる
        conn, addr = s.accept()
        with conn:
            while True:
                # データを受け取る
                data = None
                data = conn.recv(100000)
                obj = pickle.loads(data)
                if not data:
                    break
                dt_now = datetime.datetime.now()
                print(len(obj)/4)
                graph, names = graph_utils.convertData2graph(
                    obj, 10000, include_names=False)
                if graph is not None:
                    graph_utils.visualize_graph(graph, node_labels=names,
                                    save_graph_name=None, show_graph=True)
                    probability = cf.classificate(graph)
                    print(probability)
                    send_data = pickle.dumps(probability, protocol=2)
                # クライアントにデータを返す(byte でないといけない)
                conn.sendall(send_data)
