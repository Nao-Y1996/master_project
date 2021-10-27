# socket サーバを作成

import socket
import time
import datetime


# AF = IPv4 という意味
# TCP/IP の場合は、SOCK_STREAM を使う
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # IPアドレスとポートを指定
    s.bind(('192.168.0.106', 50007))
    # 1 接続
    s.listen(1)
    # connection するまで待つ
    while True:
        # 誰かがアクセスしてきたら、コネクションとアドレスを入れる
        conn, addr = s.accept()
        with conn:
            while True:
                # データを受け取る
                data = conn.recv(1024)
                if not data:
                    break
                dt_now = datetime.datetime.now()
                print(f"{dt_now.hour}'{dt_now.minute}'{dt_now.second}=> data : {data}{addr}\n")
                # クライアントにデータを返す(b -> byte でないといけない)
                conn.sendall(b'received on Mac'+ data)
