#!/usr/bin/python
# -*- coding: utf-8 -*-
# クライアントを作成

import pickle
import socket
import time
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     # サーバを指定
#     s.connect(('127.0.0.1', 50007))
#     # サーバにメッセージを送る
#     s.sendall(b'hello')
#     # ネットワークのバッファサイズは1024。サーバからの文字列を取得する
#     data = s.recv(1024)
#     #
#     print(repr(data))

# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #オブジェクトの作成をします

# while True:
#     try:
#         client.connect(('192.168.0.106', 7010)) #これでサーバーに接続します
        
#         data = pickle.dumps([1,2,3,4])
#         client.send(data) #適当なデータを送信します（届く側にわかるように）
#     except:
#         print('--')
#         pass

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #オブジェクトの作成をします
client.connect(('192.168.0.101', 50008)) #これでサーバーに接続します
for i in range(10000):
    data = pickle.dumps([i, i+1, i+2])
    client.send(data) #適当なデータを送信します（届く側にわかるように）
    print(i)
    time.sleep(0.1)
    