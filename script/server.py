#!/usr/bin/python3
# -*- coding: utf-8 -*-

import socket
import datetime
import pickle
from graph_converter import convertData2graph, visualize_graph
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

from classificator_gcn import classificator
print("start-----------------------------------")
# AF = IPv4 という意味
# TCP/IP の場合は、SOCK_STREAM を使う
cf = classificator()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # IPアドレスとポートを指定
    s.bind(('192.168.2.102', 50010))
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
                graph, names = convertData2graph(
                    obj, 10000, include_names=False)
                if graph is not None:
                    visualize_graph(graph, node_labels=names,
                                    save_graph_name=None, show_graph=True)
                    probability = cf.classificate(graph)
                    print(probability)
                    send_data = pickle.dumps(probability, protocol=2)

                # print(
                #     f"{dt_now.hour}'{dt_now.minute}'{dt_now.second}=> data : {obj}{addr}\n")
                # クライアントにデータを返す(b -> byte でないといけない)
                conn.sendall(send_data)
