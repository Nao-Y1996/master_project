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


# class ProbabilitySbscriber(object):

#     def __init__(self, topic_name):
#         self.probability = None
#         self._probability_sub = rospy.Subscriber(topic_name, Float32MultiArray, self.callback)
#         rospy.wait_for_message(topic_name, Float32MultiArray, timeout=5.0)

#     def callback(self, data):
#         self.probability = data.data
    
#     def get_probability(self):
#         return  self.probability

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

    # probability_sub = ProbabilitySbscriber(topic_name='probability')

    # 認識結果（probability）を受け取るための通信の設定
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0]
    port = 5624
    sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM) # ソケットを作成する
    sock.bind((IP_ADDRESS, port)) # 使用するIPアドレスとポート番号を指定
    print(f'Probability Subscriber : IP address = {IP_ADDRESS}  port = {port}')


    # 認識の確率表示のグラフ設定
    # labels = []
    # save_dir = rospy.get_param("/save_dir")
    # read_data = pd.read_csv(save_dir +'/state.csv',encoding="utf-8")
    # labels = read_data['state'].tolist()
    labels = ['work', 'eating', 'reading']
    fig, ax = plt.subplots()

    data_buf_len = 10
    pattern_num = len(labels)
    count = 0
    probability_list = np.array([[0.0]*pattern_num] * data_buf_len)
    flag_display = False
    while not rospy.is_shutdown():
        # robot_mode = rospy.get_param("/robot_mode")


        # probabilityをUDPで受け取る
        # probability = probability_sub.get_probability()
        print('------------------------------------------------------------')
        probability, cli_addr = sock.recvfrom(1024)
        probability = pickle.loads(probability)


        # 認識確率の表示
        probability_list[count] = probability
        display_probability  = probability_list.mean(axis=0)
        if flag_display:
            show_probability_graph(ax, labels, np.round(display_probability, decimals=4).tolist())
        count += 1
        if count >= data_buf_len:
            flag_display = True
            count = 0


        spin_rate.sleep()