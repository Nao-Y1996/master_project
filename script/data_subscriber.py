#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import socket
import pickle
from graph_tools import graph_utilitys
from classificator_nnconv import classificator
import traceback
import json


graph_utils = graph_utilitys(fasttext_model=os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin')
detectable_obj_num = len(graph_utils.ID_2_OBJECT_NAME.keys())
all_obj_names = graph_utils.ID_2_OBJECT_NAME.values()


class DataSubscriber():
    def __init__(self):
        self.observed_data = [0.0, 0.0, 0.0, 0.0]
        self.data_sub = rospy.Subscriber("/observed_data", Float32MultiArray, self.callback)
        # rospy.wait_for_message("/observed_data", Float32MultiArray, timeout=10.0)

    def callback(self, data):
        self.observed_data = data.data

    def get_data(self):
        return self.observed_data


if __name__ == '__main__':

    rospy.init_node('model_nnconv', anonymous=True)
    spin_rate=rospy.Rate(20)

    user_name = rospy.get_param("/user_name")

    # -------------------------- 認識モデルの初期設定 --------------------------
    model_path = '/home/'+ os.getlogin() +'/catkin_ws/src/master_project/script/recognition_model/'+user_name+'_model.pt'
    model_info_path = '/home/'+ os.getlogin() +'/catkin_ws/src/master_project/script/recognition_model/'+user_name+'_model_info.json'
    model_update_time, modelinfo_update_time = None, None
    model_exist = os.path.exists(model_path) and os.path.exists(model_info_path)
    if model_exist:
        model_update_time = os.path.getmtime(model_path)
        modelinfo_update_time = os.path.getmtime(model_info_path)
        print('モデルの初期設定を読み込む')

        with open(model_info_path) as f:
            _dict = json.load(f)
            pattern_num = _dict['pattern_num']
        cf = classificator(model=model_path, output_dim=pattern_num)
        
        data_buf_len = 10
        count4probability = 0
        probability_list = np.array([[0.0]*pattern_num] * data_buf_len)

        time_window = 10
        is_unnecessary_obj_list = np.array([[0.0]*detectable_obj_num] * time_window) # 過去time_windowフレーム分の不要物体の情報を格納するリスト（0=必要物体、1=不要物体）

        model_loaded = True
    else:
        print('まだモデルが存在しない')
        model_loaded = False
    # -------------------------------------------------------------------- 

# ------------------------データを受け取るための設定--------------------
    data_sub = DataSubscriber()

# ----------------------------------------------------------------------
    
# ------------------------認識確率の配信のための設定------------------------
    probability_pub = rospy.Publisher("avarage_probability", Float32MultiArray, queue_size=1)
# ----------------------------------------------------------------------

# ------------------------片付け物体情報の配信のための設定------------------------
    cleaninfo_pub = rospy.Publisher("cleaninfo", Float32MultiArray, queue_size=1)
# ----------------------------------------------------------------------



    frame_count = 0
    while not rospy.is_shutdown():
        robot_mode = rospy.get_param("/robot_mode")
        clean_mode = rospy.get_param("/is_clean_mode")
        model_exist = os.path.exists(model_path) and os.path.exists(model_info_path)
        if model_exist:
            # --------------------- モデルが更新されたらモデルの設定を読み込み、各種変数を初期化する ---------------------
            if model_update_time != os.path.getmtime(model_path) and modelinfo_update_time != os.path.getmtime(model_info_path):
                model_update_time = os.path.getmtime(model_path)
                modelinfo_update_time = os.path.getmtime(model_info_path)
                print('モデルが更新された')

                with open(model_info_path) as f:
                    _dict = json.load(f)
                    pattern_num = _dict['pattern_num']
                    model_name = _dict['model_name']
                cf = classificator(model=model_path, output_dim=pattern_num)
                
                data_buf_len = 10
                count4probability = 0
                probability_list = np.array([[0.0]*pattern_num] * data_buf_len)

                time_window = 10
                is_unnecessary_obj_list = np.array([[0.0]*detectable_obj_num] * time_window) # 過去time_windowフレーム分の不要物体の情報を格納するリスト（0=必要物体、1=不要物体）

                model_loaded = True
            else:
                model_loaded = True
        else:
            model_loaded = False
        # ---------------------------------------------------------------------------------------------------
        if model_loaded:
            if robot_mode == 'state_recognition':
                
                # データを受け取る
                data = data_sub.get_data()
                
                # グラフ形式に変換
                position_data = graph_utils.removeDataId(data)
                graph, node_names = graph_utils.positionData2graph(position_data, 10000, include_names=True)
                if graph is not None:
                    # graph_utils.visualize_graph(graph, node_labels=node_names, save_graph_name=None, show_graph=True) # 状態グラフの表示

                    # 状態認識
                    probability = cf.classificate(graph)

                    # 認識確率の平均（過去data_buf_len個分のデータで平均を取る）
                    probability_list[count4probability] = probability
                    average_probability  = probability_list.mean(axis=0).tolist()
                    count4probability += 1
                    if count4probability >= data_buf_len:
                        count4probability = 0

                    # 不要な物体（ノード）の特定
                    if clean_mode:
                        is_unnecessary_obj_list[frame_count] = [0.0]*detectable_obj_num
                        
                        # ノードを１つ取り除いたパターンのグラフを取得
                        dummy_graph_lsit, removed_obj_data_list = graph_utils.convertData2dummygraphs(data)

                        unnecessary_obj_candidate_info = [] # 物体ノードを一つ取り除いたとき、認識ラベルが一致かつ確率が上昇する物体の情報を格納するリスト
                        for dummy_graph, removed_obj_data in zip(dummy_graph_lsit, removed_obj_data_list):
                            removed_obj_id = removed_obj_data[0]
                            if removed_obj_id == 0:
                                continue # faceは片付け対象としない

                            if dummy_graph[0] is not None:
                                removed_obj = graph_utils.ID_2_OBJECT_NAME[int(removed_obj_id)]
                                dummy_probability = cf.classificate(dummy_graph[0])
                                # あるノードを取り除いた時の認識結果ともとの認識結果が一致するか
                                if dummy_probability.index(max(dummy_probability)) == average_probability.index(max(average_probability)):
                                    # あるノードを取り除いた時の認識結果の確率が上昇するか
                                    if max(dummy_probability) > max(average_probability):
                                        # print('dummy graph : ', dummy_probability)
                                        # print('original graph : ', probability)
                                        diff =  max(dummy_probability) - max(average_probability)
                                        # print('diff : ', diff)
                                        unnecessary_obj_candidate_info.append([removed_obj_id, average_probability, dummy_probability, diff])
                                        # print(np.array(unnecessary_obj_candidate_info))
                                    else:
                                        # print('確率は上昇しませんでした')
                                        pass
                                else:
                                    # print('認識結果が一致していません')
                                    pass
                                
                        unnecessary_obj_candidate_info = np.array(unnecessary_obj_candidate_info)
                        try:
                            unnecessary_obj_index = np.argmax(unnecessary_obj_candidate_info[:,-1]) # 確率の上昇が最も大きい物体（不要物体）のインデックスを取得
                            unnecessary_obj_id = unnecessary_obj_candidate_info[unnecessary_obj_index][0] # 不要物体のIDを取得
                            unnecessary_obj_diff = unnecessary_obj_candidate_info[unnecessary_obj_index][-1] # 不要物体の確率の上昇幅を取得
                            unnecessary_obj = graph_utils.ID_2_OBJECT_NAME[int(unnecessary_obj_id)] #不要物体の名前を取得
                            print(unnecessary_obj,unnecessary_obj_diff)
                            is_unnecessary_obj_list[frame_count][int(unnecessary_obj_id)] = 1 # 現在のフレームにおける不要物体を記録
                        except IndexError:
                            pass
                        except:
                            traceback.print_exc()
                    else: # not clean_mode
                        is_unnecessary_obj_list[frame_count] = [0.0]*detectable_obj_num

                else: # graph is None
                    is_unnecessary_obj_list[frame_count] = [0.0]*detectable_obj_num
                    average_probability = [0.0] * pattern_num
                    pass
            
            else:
                average_probability = [0.0] * pattern_num
                pass

            #　認識結果（確率）を配信
            msg_average_probability = Float32MultiArray(data=average_probability)
            probability_pub.publish(msg_average_probability)

            # 片付け物体情報を配信
            unnecessary_score_list = is_unnecessary_obj_list.mean(axis=0).tolist()
            msg_cleaninfo = Float32MultiArray(data=unnecessary_score_list)
            cleaninfo_pub.publish(msg_cleaninfo)

            frame_count += 1
            if frame_count == time_window:
                frame_count = 0
        else:
            pass

        spin_rate.sleep()