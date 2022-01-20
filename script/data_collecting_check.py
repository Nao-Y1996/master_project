#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
import socket
import pickle

if __name__ == '__main__':

    rospy.init_node('collecting_check', anonymous=True)
    spin_rate=rospy.Rate(20)

    # dataを受け取るための通信の設定
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0]
    port1 = 54321
    port2 = 56789
    locaddr1 = (IP_ADDRESS, port1)
    locaddr2 = (IP_ADDRESS, port2)

    print(f'count saved : IP address = {IP_ADDRESS}  port = {port1}')
    sock1 = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM) # ソケットを作成する
    sock1.bind(locaddr1) # 使用するIPアドレスとポート番号を指定


    print(f'object names : IP address = {IP_ADDRESS}  port = {port2}')
    sock2 = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM) # ソケットを作成する
    sock2.bind(locaddr2) # 使用するIPアドレスとポート番号を指定


    while not rospy.is_shutdown():
        
        # dataをUDPでデータを受け取る
        print('------------------------------------------------------------')
        count_saved, cli_addr = sock1.recvfrom(1024)
        count_saved = pickle.loads(count_saved)

        obj_names, cli_addr = sock2.recvfrom(1024)
        obj_names = pickle.loads(obj_names)

        print(count_saved, obj_names)


        spin_rate.sleep()