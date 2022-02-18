#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt
import socket
import pickle


def show_probability_graph(ax, labels, probability, user_name):
    x = np.arange(len(labels))
    width = 0.35
    rects = ax.bar(x, probability, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.title('state pattern of ' + user_name)
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
    save_dir = rospy.get_param("/save_dir")
    # read_data = pd.read_csv(save_dir +'/state.csv',encoding="utf-8")
    # labels = read_data['state'].tolist()
    user_name = rospy.get_param('user_name')
    
    fig, ax = plt.subplots()
    while not rospy.is_shutdown():
        robot_mode = rospy.get_param('robot_mode')

        if robot_mode=='state_recognition':
            # probabilityをUDPで受け取る
            data, cli_addr = sock.recvfrom(256)
            average_probability = pickle.loads(data)
            print(average_probability)
            labels = ['state'+str(i) for i in range(len(average_probability))]
            # 認識確率の表示
            show_probability_graph(ax, labels, np.round(average_probability, decimals=4).tolist(), user_name)
        else:
            # average_probability = [0.0, 0.0, 0.0]
            pass
        

        spin_rate.sleep()