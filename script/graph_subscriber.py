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
graph_utils = graph_utilitys(fasttext_model=os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin')


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

    rospy.init_node('model_nnconv', anonymous=True)
    spin_rate=rospy.Rate(20)

    # dataを受け取るための通信の設定
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0]
    port = 12345
    locaddr = (IP_ADDRESS, port)
    sock4data = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM) # ソケットを作成する
    sock4data.bind(locaddr) # 使用するIPアドレスとポート番号を指定
    print(f'data server : IP address = {IP_ADDRESS}  port = {port}')

    # 認識モデルの設定
    user_dir = rospy.get_param("/user_dir").replace('kubotalab-hsr', os.getlogin())
    model_path = user_dir+'/model_nnconv.pt'
    cf = classificator(model=model_path)

    # 認識確率送信のためのソケットを作成する（UDP）
    sock4probability = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serv_address = ('192.168.0.110', 5624)

    # 認識の確率表示のグラフ設定
    labels = ['working', 'eating', 'reading']
    fig, ax = plt.subplots()

    # 
    data_buf_len = 10
    pattern_num = len(labels)
    count = 0
    probability_list = np.array([[0.0]*pattern_num] * data_buf_len)
    flag_display = False
    while not rospy.is_shutdown():
        robot_mode = rospy.get_param("/robot_mode")
        clean_mode = rospy.get_param("/is_clean_mode")
        
        # dataをUDPでデータを受け取る
        print('------------------------------------------------------------')
        data, cli_addr = sock4data.recvfrom(1024)
        data = pickle.loads(data)
        
        # グラフ形式に変換
        position_data = graph_utils.removeDataId(data)
        graph, node_names = graph_utils.positionData2graph(position_data, 10000, include_names=True)
        
        if graph is not None:
            # graph_utils.visualize_graph(graph, node_labels=node_names, save_graph_name=None, show_graph=True) # 状態グラフの表示
            
            # 状態認識
            if robot_mode == 'state_recognition':
                probability = cf.classificate(graph)

                # 認識確率の平滑化
                probability_list[count] = probability
                average_probability  = probability_list.mean(axis=0).tolist()
                count += 1
                if count >= data_buf_len:
                    count = 0
                state_now = labels[average_probability.index(max(average_probability))]
                if not clean_mode:
                    print(f'状態 : {state_now}')
                    print(f'確率 : {average_probability}')
                
                #------------ 認識結果をUDPで送信 ------------#
                send_len = sock4probability.sendto(pickle.dumps(average_probability, protocol=2), serv_address)
                # 受け取る側がpython2の場合はprotocol=2を指定する
                #----------------------------------------------#

                # 認識確率の表示
                # show_probability_graph(ax, labels, np.round(average_probability, decimals=4).tolist())

                # 不要な物体（ノード）の特定
                if clean_mode:
                    # ノードを１つ取り除いたパターンのグラフを取得
                    dummy_graph_lsit, removed_obj_data_list = graph_utils.convertData2dummygraphs(data)

                    unnecessary_obj_candidate_info = []
                    for dummy_graph, removed_obj_data in zip(dummy_graph_lsit, removed_obj_data_list):
                        removed_obj_id = removed_obj_data[0]
                        if removed_obj_id == 0:
                            continue # faceは片付け対象としない

                        if dummy_graph[0] is not None:
                            removed_obj = graph_utils.ID_2_OBJECT_NAME[int(removed_obj_id)]
                            # print()
                            # print(f'   removed  : {removed_obj}')
                            dummy_probability = cf.classificate(dummy_graph[0])
                            # print(f'probability : {dummy_probability}')
                            # あるノードを取り除いた時の認識結果ともとの認識結果が一致するか
                            if dummy_probability.index(max(dummy_probability)) == average_probability.index(max(average_probability)):
                                # あるノードを取り除いた時の認識結果の確率が上昇するか
                                if max(dummy_probability) > max(average_probability):
                                    # print('dummy graph : ', dummy_probability)
                                    # print(' true graph : ', probability)
                                    diff =  max(dummy_probability) - max(average_probability)
                                    # print('diff : ', diff)
                                    unnecessary_obj_candidate_info.append([removed_obj_id, average_probability, dummy_probability, diff])
                                else:
                                    # print('確率は上昇しませんでした')
                                    pass
                            else:
                                # print('認識結果が一致していません')
                                pass

                    print('=======不要ノード========')
                    unnecessary_obj_candidate_info = np.array(unnecessary_obj_candidate_info)
                    try:
                        # for id, diff in zip(unnecessary_obj_candidate_info[:,0], list(unnecessary_obj_candidate_info[:,-1])):
                        #     print(graph_utils.ID_2_OBJECT_NAME[int(id)], diff)
                        unnecessary_obj_index = np.argmax(unnecessary_obj_candidate_info[:,-1])
                        unnecessary_obj_id = unnecessary_obj_candidate_info[unnecessary_obj_index][0]
                        unnecessary_obj = graph_utils.ID_2_OBJECT_NAME[int(unnecessary_obj_id)]
                        print(unnecessary_obj)
                    except IndexError:
                        pass
                    except:
                        traceback.print_exc()
                    print('=======================')
                else:
                    pass
                 
        spin_rate.sleep()