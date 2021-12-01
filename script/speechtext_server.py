#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import time

M_SIZE = 1024

# 
host = '127.0.0.1'
port = 8890

locaddr = (host, port)

# ①ソケットを作成する
sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
print('Successfully created socket!')

# ②使用するIPアドレスとポート番号を指定
sock.bind(locaddr)

while True:
    try :
        # ③Clientからのmessageの受付開始
        message, cli_addr = sock.recvfrom(M_SIZE)
        message = message.decode(encoding='utf-8')
        print(f'Received message is [{message}]')
        if '覚えて' in message:
            print('GNN学習データの収集を開始します')
            # ここでデータ収集ノードを立ち上げ
        else:
            pass
            # print(f'Received message is [{message}]')

    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close()
        break