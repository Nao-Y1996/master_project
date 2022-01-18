#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import socket
import pickle
from graph_converter import graph_utilitys
import matplotlib.pyplot as plt
from classificator_nnconv import classificator
import traceback
graph_utils = graph_utilitys(fasttext_model='cc.en.300.bin')


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

    rospy.init_node('master_model_nnconv', anonymous=True)
    spin_rate=rospy.Rate(10)

    # dataを受け取るための通信の設定
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0]
    port = 12345
    locaddr = (IP_ADDRESS, port)
    sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM) # ソケットを作成する
    sock.bind(locaddr) # 使用するIPアドレスとポート番号を指定
    print(f'data server : IP address = {IP_ADDRESS}  port = {port}')

    # 認識モデルの設定
    cf = classificator(model=os.path.dirname(os.path.abspath(__file__))+ '/model/master_model_nnconv.pt')

    # 認識確率の配信用
    probability_pub = rospy.Publisher('probability', Float32MultiArray, queue_size=1)

    # 認識の確率表示のグラフ設定
    labels = ['eating', 'work', 'rest', 'reading']
    fig, ax = plt.subplots()

    while not rospy.is_shutdown():
        robot_mode = rospy.get_param("/robot_mode")
        clean_mode = rospy.get_param("/is_clean_mode")
        
        # dataをUDPでデータを受け取る
        print('------------------------------------------------------------')
        data, cli_addr = sock.recvfrom(1024)
        data = pickle.loads(data)

        # グラフ形式に変換
        position_data = graph_utils.removeDataId(data)
        graph, node_names = graph_utils.positionData2graph(position_data, 10000, include_names=True)
        
        # ノードを１つ取り除いたパターンのグラフを取得
        dummy_graph_lsit, removed_obj_data_list = graph_utils.convertData2dummygraphs(data)

        if graph is not None:
            # graph_utils.visualize_graph(graph, node_labels=node_names, save_graph_name=None, show_graph=True) # 状態グラフの表示
            
            # 状態認識
            if robot_mode == 'state_recognition':
                probability = cf.classificate(graph)
                # print(probability)
                show_probability_graph(ax, labels, np.round(probability, decimals=4).tolist())

                # 不要な物体（ノード）の特定
                if clean_mode:
                    for dummy_graph, removed_obj_data in zip(dummy_graph_lsit, removed_obj_data_list):
                        removed_obj_id = removed_obj_data[0]
                        unnecessary_obj = graph_utils.ID_2_OBJECT_NAME[int(removed_obj_id)]

                        if dummy_graph[0] is not None:
                            dummy_probability = cf.classificate(dummy_graph[0])
                            # あるノードを取り除いた時の認識結果ともとの認識結果が一致するか
                            if dummy_probability.index(max(dummy_probability)) == probability.index(max(probability)):
                                print(probability)
                                print(dummy_probability)
                                # あるノードを取り除いた時の認識結果の確率が上昇するか
                                if max(dummy_probability) > max(probability):
                                    print('=======不要ノード========')
                                    # print('dummy : ', dummy_probability)
                                    # print('graph : ', probability)
                                    print('diff : ',  max(dummy_probability) - max(probability))
                                    print(unnecessary_obj)
                                    print('=======================')
                                else:
                                    # print('確率は上昇しませんでした')
                                    pass
                            else:
                                # print('認識結果が一致していません')
                                pass
                else:
                    pass

                try:
                    publish_data = Float32MultiArray(data=probability)
                    probability_pub.publish(publish_data)
                except rospy.exceptions.ROSSerializationException:
                    # 認識モードから通常モードへの切替時に
                    # rospy.exceptions.ROSSerializationException: field data[] must be float type
                    # のエラーが出るのでそれ用
                    continue
                except:
                    traceback.print_exc()
        spin_rate.sleep()