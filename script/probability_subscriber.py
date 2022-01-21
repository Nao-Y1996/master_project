#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import socket
import pickle


def show_probability_graph(ax, labels, probability):
    x = np.arange(len(labels))
    width = 0.35
    rects = ax.bar(x, probability, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.ylim(0, 1)
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.draw()  # 描画する。
    plt.pause(0.01)  # 0.01 秒ストップする。
    plt.cla()


if __name__ == '__main__':

    rospy.init_node('probability_subscriber', anonymous=True)
    spin_rate=rospy.Rate(10)


    # 認識結果（probability）を受け取るための通信の設定
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0]
    port = 5624
    sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM) # ソケットを作成する
    sock.bind((IP_ADDRESS, port)) # 使用するIPアドレスとポート番号を指定
    print('Probability Subscriber : IP address = ' + str(IP_ADDRESS) + ' port = ' + str(port))



    # 認識の確率表示のグラフ設定
    # labels = []
    # save_dir = rospy.get_param("/save_dir")
    # read_data = pd.read_csv(save_dir +'/state.csv',encoding="utf-8")
    # labels = read_data['state'].tolist()
    labels = ['working', 'eating', 'reading']
    fig, ax = plt.subplots()

    pattern_num = len(labels)
    while not rospy.is_shutdown():

        # probabilityをUDPで受け取る
        print('------------------------------------------------------------')
        average_probability, cli_addr = sock.recvfrom(1024)
        average_probability = pickle.loads(average_probability)

        # 認識確率の表示
        show_probability_graph(ax, labels, np.round(average_probability, decimals=4).tolist())

        spin_rate.sleep()