#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt
import socket
import pickle
import os
import matplotlib.pyplot as plt

all_obj_names = rospy.get_param("/all_obj_names")
detectable_obj_num = len(all_obj_names)
# print(all_obj_names)
# print(type(all_obj_names))


def show_cleaninfo_graph(ax, labels, probability):
    x = np.arange(len(labels))
    width = 0.35
    rects = ax.bar(x, probability, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=75)
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

    rospy.init_node('clean_info_subscriber', anonymous=True)
    spin_rate=rospy.Rate(10)


    # cleaninfoを受け取るための通信の設定
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0]
    port = 3456
    sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM) # ソケットを作成する
    sock.bind((IP_ADDRESS, port)) # 使用するIPアドレスとポート番号を指定
    print('cleaninfo Subscriber : IP address = ' + str(IP_ADDRESS) + ' port = ' + str(port))



    # 認識の確率表示のグラフ設定
    # labels = []
    # save_dir = rospy.get_param("/save_dir")
    # read_data = pd.read_csv(save_dir +'/state.csv',encoding="utf-8")
    # labels = read_data['state'].tolist()
    # labels = ['working', 'eating', 'reading']
    fig, ax = plt.subplots()
    # pattern_num = len(labels)
    while not rospy.is_shutdown():
        clean_mode = rospy.get_param("/is_clean_mode")

        if clean_mode:
            data, cli_addr = sock.recvfrom(256)
            data = pickle.loads(data)
        else:
            data = [0.0] * detectable_obj_num

        # 認識確率の表示
        show_cleaninfo_graph(ax, all_obj_names, np.round(data, decimals=4).tolist())
        # if data[np.argmax(data)] == 1.0:
        #     rospy.set_param('/clean_obj_id', np.argmax(data))
        # else:
        #     pass


        spin_rate.sleep()