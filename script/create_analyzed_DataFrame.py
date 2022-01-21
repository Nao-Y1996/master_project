#!/usr/bin/python3
# -*- coding: utf-8 -*-
# import rospy
# from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import socket
import pickle
from graph_converter import graph_utilitys
import matplotlib.pyplot as plt
from classificator_nnconv import classificator
import traceback
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd
import time


def show_probability_graph(ax, labels, probability, count=None, is_save=False):
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
    if is_save:
        fig.savefig('fig'+str(count)+'.png')



if __name__ == '__main__':
    ft_path = os.path.dirname(__file__) +'/w2v_model/cc.en.300.bin'
    graph_utils = graph_utilitys(fasttext_model=ft_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/2022-01-20/user_1"

    
    # 認識モデルの設定
    model_path = base_dir+ '/master_model_nnconv1.pt'
    cf = classificator(model=model_path)
    
    # 読み込むデータ
    data_dir = base_dir+ '/position_data'
    csv_path_dict = {0:data_dir+'/pattern_0.csv',1:data_dir+'/pattern_1.csv',2:data_dir+'/pattern_2.csv'}

    # 認識の確率表示のグラフ設定
    labels = ['work', 'eating', 'reading']
    # fig, ax = plt.subplots()
    
    # 3次元グラフの表示
    show_3dgraph = False
    fig3d = plt.figure()
    ax3d = Axes3D(fig3d)

    # 平滑化のための設定
    # data_buf_len = 10
    # pattern_num = len(labels)
    
    # probability_list = np.array([[0.0]*pattern_num] * data_buf_len)
    # flag_display = False
    
    # while not rospy.is_shutdown():
    for true_label, csv_file_name in csv_path_dict.items():
        columun_names= ['state', 'probability', "objects", 'removed_obj', 'dummy_state', 'dummy_probability', 'state_match', 'run_up', 'diff', 'is_unnecessary']
        df = pd.DataFrame(columns=columun_names,)
        analyze_data = [None]*len(columun_names)
        # csvファイルの読み込み
        count = 0
        with open(csv_file_name) as f:
            csv_file = csv.reader(f)
            # dataを1ずつ読み込む
            for i, _row in enumerate(csv_file):
                
                data = []
                if '' in _row:
                    continue
                for j, v in enumerate(_row):
                    data.append(float(v))
                # dataをグラフ形式に変換
                data_id = data[0]
                position_data = graph_utils.removeDataId(data)
                graph, node_names = graph_utils.positionData2graph(position_data, 10000, include_names=True)
                
                # グラフの枠を作成用
                position_data = np.reshape(position_data, (-1,4))
                
                if graph is not None:
                    # graph_utils.visualize_graph(graph, node_labels=node_names, save_graph_name=None, show_graph=True) # 状態グラフの表示
                    
                    # 状態認識
                    probability = cf.classificate(graph)
                    average_probability  = probability
                    
                    if show_3dgraph:
                        obj_ids = position_data[:,0]
                        position_data_x = position_data[:,1]
                        position_data_y = position_data[:,2]
                        position_data_z = position_data[:,3]
                        # ax3d.set_xlabel('X ')
                        # ax3d.set_ylabel('Y ')
                        # ax3d.set_zlabel('Z ')
                        ax3d.set_xlim(-0.5, 0.5)
                        ax3d.set_ylim(-0.5, 0.5)
                        ax3d.set_zlim(0.9, 1.7)
                        ax3d.set_title(str(np.round(probability, -4)))
                        for obj_id, x,y,z in zip(obj_ids, position_data_x, position_data_y,position_data_z):
                            name = graph_utils.ID_2_OBJECT_NAME[int(obj_id)]
                            colorDict = {'face':"k", 'tvmonitor':"g", 'laptop':"b", 'mouse':"c", 'keyboard':"m", 'sandwitch':"y", 'salada':"r", 'soup':'#ff7f00','book':'#e41a1c'}
                            try:
                                color = colorDict[name]
                            except:
                                color = '#377eb8'
                            ax3d.scatter(x, y, z, c=color, marker='o', s=100, label=name)
                            ax3d.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
                        # ax3d.plot(position_data_x,position_data_y,position_data_z,marker="o",linestyle='None')
                        plt.draw()  # 描画する。
                        plt.pause(0.01)  # 0.01 秒ストップする。
                        plt.cla()
                    
                    # 認識確率の表示
                    # show_probability_graph(ax, labels, np.round(average_probability, decimals=4).tolist())

                    # 不要な物体（ノード）の特定
                    state_now = labels[average_probability.index(max(average_probability))]
                    print(f'===================================== {data_id} ===========================================')
                    # print(f'状態 : {state_now}')
                    # print(f'確率 : {average_probability}')
                    
                    analyze_data[0] = state_now
                    analyze_data[1] = probability
                    analyze_data[2] = node_names
                    
                    
                    # ノードを１つ取り除いたパターンのグラフを取得
                    dummy_graph_lsit, removed_obj_data_list = graph_utils.convertData2dummygraphs(data)

                    unnecessary_obj_candidate_info = []
                    for dummy_graph, removed_obj_data in zip(dummy_graph_lsit, removed_obj_data_list):
                        removed_obj_id = removed_obj_data[0]
                        if removed_obj_id == 0:
                            continue # faceは片付け対象としない

                        if dummy_graph[0] is not None:
                            _countList = []
                            # print('------------------------------')
                            removed_obj = graph_utils.ID_2_OBJECT_NAME[int(removed_obj_id)]
                            # print(f'{count}    removed : {removed_obj}')
                            dummy_probability = cf.classificate(dummy_graph[0])
                            dummy_state = labels[dummy_probability.index(max(dummy_probability))]
                            # print(dummy_probability)
                            state_match = False
                            run_up = False
                            diff = None
                            # あるノードを取り除いた時の認識結果ともとの認識結果が一致するか
                            if dummy_probability.index(max(dummy_probability)) == average_probability.index(max(average_probability)):
                                state_match = True
                                # あるノードを取り除いた時の認識結果の確率が上昇するか
                                if max(dummy_probability) > max(average_probability):
                                    run_up = True
                                    diff =  (max(dummy_probability) - max(average_probability)) * 100
                                    # print(f'diff : {diff}')
                                    unnecessary_obj_candidate_info.append([removed_obj_id, average_probability, dummy_probability, diff, count])
                                else:
                                    # print('確率は上昇せず')
                                    run_up = False
                                    pass
                            else:
                                # print('認識結果が一致していません')
                                # print(f'認識が変化 : {state_now} --> {dummy_state}')
                                state_match = False
                                pass
                            
                            analyze_data[3] = removed_obj
                            analyze_data[4] = dummy_state
                            analyze_data[5] = dummy_probability
                            analyze_data[6] = state_match
                            analyze_data[7] = run_up
                            analyze_data[8] = diff
                            analyze_data[9] = False
                            df.loc[count] = analyze_data
                            count += 1
                                

                    # print('~~~~~~~~~不要ノード~~~~~~~~~')
                    unnecessary_obj_candidate_info = np.array(unnecessary_obj_candidate_info)
                    # for _id, diff, _count in zip(unnecessary_obj_candidate_info[:,0], unnecessary_obj_candidate_info[:,-1], _countList):
                    #     print(graph_utils.ID_2_OBJECT_NAME[int(_id)], diff)
                    unnecessary_obj_index = np.argmax(unnecessary_obj_candidate_info[:,-2])
                    unnecessary_obj_count = unnecessary_obj_candidate_info[unnecessary_obj_index][-1]
                    unnecessary_obj_id = unnecessary_obj_candidate_info[unnecessary_obj_index][0]
                    unnecessary_obj = graph_utils.ID_2_OBJECT_NAME[int(unnecessary_obj_id)]
                    # print(unnecessary_obj_count, unnecessary_obj)
                    df.at[unnecessary_obj_count, 'is_unnecessary'] = True
                    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                
        df.to_csv(csv_file_name.replace('.csv', '_analyzed.csv'))
