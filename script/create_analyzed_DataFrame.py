#!/usr/bin/python3
# -*- coding: utf-8 -*-
# import rospy
# from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import socket
import pickle
from graph_tools import graph_utilitys
import matplotlib.pyplot as plt
from classificator_nnconv import classificator
import traceback
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd
import sys


# 名前の変更パターン
all_pattern = {0:{'bottle':'toothbrush'},
            1:{'bottle':'scissors'},
            2:{'bottle':'tape'},
            3:{'bottle':'can'},
            4:{'bottle':'wallet'},
            5:{'bottle':'cap'},
            6:{'bottle':'hairdryer'}}

if __name__ == '__main__':
    user_name = input('enter user name\n')
    model_name = input('enter gnn model name (ex:abc.pt)\n')
    base_dir = '/home/'+ os.getlogin() +'/catkin_ws/src/master_project/script/'
    user_dir = base_dir+ "experiment_data/"+user_name
    model_path = user_dir+'/learning_outputs/'+model_name
    if os.path.exists(model_path):
        pass
    else:
        sys.exit('モデルが存在しません')
    cf = classificator(model=model_path)# 認識モデルの設定

    for k, pattern in enumerate(all_pattern):
        print(f" ------------   {all_pattern[k]['bottle']}   --------------")
        
    
        # インスタンスを作り直す（ID_2_OBJECT_NAMEをリセット）
        graph_utils = graph_utilitys(fasttext_model=base_dir+'/w2v_model/cc.en.300.bin')
         # 認識物体の名前を変更
        graph_utils.changeID_2_OBJECT_NAME(all_pattern[k])
        # print(graph_utils.ID_2_OBJECT_NAME)

        # 読み込むデータ
        data_dir = user_dir+ '/position_data'
        csv_path_dict = {0:data_dir+'/bottle_augmented_pattern_0.csv',1:data_dir+'/bottle_augmented_pattern_1.csv',2:data_dir+'/bottle_augmented_pattern_2.csv'}

        # 認識の確率表示のグラフ設定
        labels = ['working', 'eating', 'reading']
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
            # print('-------------------------------------------------')
            # print(csv_file_name.replace('bottle', all_pattern[k]['bottle']).replace('.csv', '_analyzed.csv'))
            # print('-------------------------------------------------')
            if all_pattern[k]['bottle'] =='toothbrush' and true_label==0:
                continue
            columun_names= ['dataID', 'state', 'probability', "objects", 'removed_obj', 'dummy_state', 'dummy_probability', 'state_match', 'is_probability_rised', 'diff', 'is_unnecessary']
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
                    data_id = int(data[0])
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
                        
                        # 不要な物体（ノード）の特定
                        state_now = labels[average_probability.index(max(average_probability))]
                        print(f"=================== {all_pattern[k]['bottle']} pattern_{true_label} | {data_id} {count}======================")
                        # print(f'状態 : {state_now}')
                        # print(f'確率 : {average_probability}')
                        analyze_data[0] = data_id
                        analyze_data[1] = state_now
                        analyze_data[2] = probability
                        analyze_data[3] = node_names
                        
                        
                        # ノードを１つ取り除いたパターンのグラフを取得
                        dummy_graph_lsit, removed_obj_data_list = graph_utils.convertData2dummygraphs(data)

                        unnecessary_obj_candidate_info = []
                        for dummy_graph, removed_obj_data in zip(dummy_graph_lsit, removed_obj_data_list):
                            removed_obj_id = removed_obj_data[0]
                            if removed_obj_id == 0:
                                continue # faceは片付け対象としない

                            if dummy_graph[0] is not None:
                                removed_obj = graph_utils.ID_2_OBJECT_NAME[int(removed_obj_id)]
                                dummy_probability = cf.classificate(dummy_graph[0])
                                dummy_state = labels[dummy_probability.index(max(dummy_probability))]
                                # print(dummy_probability)
                                state_match = False
                                is_probability_rised = False
                                diff = None
                                # あるノードを取り除いた時の認識結果ともとの認識結果が一致するか
                                if dummy_probability.index(max(dummy_probability)) == average_probability.index(max(average_probability)):
                                    state_match = True
                                    # あるノードを取り除いた時の認識結果の確率が上昇するか
                                    if max(dummy_probability) > max(average_probability):
                                        is_probability_rised = True
                                        diff =  (max(dummy_probability) - max(average_probability)) * 100
                                        # print(f'diff : {diff}')
                                        unnecessary_obj_candidate_info.append([average_probability, dummy_probability, removed_obj_id, diff, count])
                                    else:
                                        # print('確率は上昇しませんでした')
                                        is_probability_rised = False
                                        pass
                                else:
                                    # print('認識結果が一致していません')
                                    state_match = False
                                    pass
                                
                                analyze_data[4] = removed_obj
                                analyze_data[5] = dummy_state
                                analyze_data[6] = dummy_probability
                                analyze_data[7] = state_match
                                analyze_data[8] = is_probability_rised
                                analyze_data[9] = diff
                                analyze_data[10] = False
                                df.loc[count] = analyze_data
                                count += 1
                                    
                        if len(unnecessary_obj_candidate_info)!=0:
                            # print('~~~~~~~~~不要ノード~~~~~~~~~')
                            unnecessary_obj_candidate_info = np.array(unnecessary_obj_candidate_info)
                            # print(unnecessary_obj_candidate_info)
                            unnecessary_obj_index = np.argmax(unnecessary_obj_candidate_info[:,3])
                            # print(unnecessary_obj_index)
                            unnecessary_obj_count = unnecessary_obj_candidate_info[unnecessary_obj_index][-1]
                            unnecessary_obj_id = unnecessary_obj_candidate_info[unnecessary_obj_index][2]
                            # print(unnecessary_obj_id)
                            unnecessary_obj = graph_utils.ID_2_OBJECT_NAME[int(unnecessary_obj_id)]
                            # print(unnecessary_obj_count, unnecessary_obj)
                            df.at[unnecessary_obj_count, 'is_unnecessary'] = True
                            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        else:
                            pass
                    
            df.to_csv(csv_file_name.replace('bottle', all_pattern[k]['bottle']).replace('.csv', '_analyzed.csv'))

