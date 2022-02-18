#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
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

if __name__ == '__main__':

    rospy.init_node('model_nnconv', anonymous=True)
    spin_rate=rospy.Rate(20)

    user_name = rospy.get_param("/user_name")

# ------------------------dataを受け取るための通信の設定--------------------
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0]
    port4data = 12345
    sock4data = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
    sock4data.bind((IP_ADDRESS, port4data))
    print(f'data server : IP address = {IP_ADDRESS}  port = {port4data}')
# ----------------------------------------------------------------------
    
# ------------------------認識確率の送信のための設定------------------------
    sock4probability = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serv_address = ('192.168.0.110', 5624)
# ----------------------------------------------------------------------

# ------------------------片付け物体情報の送信のための設定------------------------
    sock4clean = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serv_address4clean = ('192.168.0.110', 3456)
# ----------------------------------------------------------------------

    # -------------------------- 認識モデルの初期設定 --------------------------
    model_path = '/home/'+ os.getlogin() +'/catkin_ws/src/master_project/script/recognition_model/'+user_name+'_model.pt'
    model_info_path = '/home/'+ os.getlogin() +'/catkin_ws/src/master_project/script/recognition_model/'+user_name+'_model_info.json'
    model_update_time, modelinfo_update_time = None, None
    if os.path.exists(model_path) and os.path.exists(model_info_path):
        model_update_time = os.path.getmtime(model_path)
        modelinfo_update_time = os.path.getmtime(model_info_path)
        print('モデルの初期設定を読み込む')

        with open(model_info_path) as f:
            _dict = json.load(f)
            pattern_num = _dict['pattern_num']
        cf = classificator(model=model_path, output_dim=pattern_num)
        
        data_buf_len = 10
        count = 0
        probability_list = np.array([[0.0]*pattern_num] * data_buf_len)

        time_window = 10
        is_unnecessary_obj_list = np.array([[0.0]*detectable_obj_num] * time_window) # 過去time_windowフレーム分の不要物体の情報を格納するリスト（0=必要物体、1=不要物体）

        model_loaded = True
    else:
        print('まだモデルが存在しない')
        model_loaded = False
    # -------------------------------------------------------------------- 

    frame_count = 0
    while not rospy.is_shutdown():
        robot_mode = rospy.get_param("/robot_mode")
        clean_mode = rospy.get_param("/is_clean_mode")
        if os.path.exists(model_path) and os.path.exists(model_info_path):
            # --------------------- モデルが更新されたらモデルの設定、各種変数を再定義する ---------------------
            if model_update_time != os.path.getmtime(model_path) and modelinfo_update_time != os.path.getmtime(model_info_path):
                model_update_time = os.path.getmtime(model_path)
                modelinfo_update_time = os.path.getmtime(model_info_path)
                print('モデルを更新した')

                with open(model_info_path) as f:
                    _dict = json.load(f)
                    pattern_num = _dict['pattern_num']
                    model_name = _dict['model_name']
                cf = classificator(model=model_path, output_dim=pattern_num)
                
                data_buf_len = 10
                count = 0
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
                
                # dataをUDPでデータを受け取る
                data, cli_addr = sock4data.recvfrom(1024)
                data = pickle.loads(data)
                
                # グラフ形式に変換
                position_data = graph_utils.removeDataId(data)
                graph, node_names = graph_utils.positionData2graph(position_data, 10000, include_names=True)
                if graph is not None:
                    # graph_utils.visualize_graph(graph, node_labels=node_names, save_graph_name=None, show_graph=True) # 状態グラフの表示

                    # 状態認識
                    probability = cf.classificate(graph)

                    # 認識確率の平滑化（過去data_buf_len個分のデータで平均を取る）
                    probability_list[count] = probability
                    average_probability  = probability_list.mean(axis=0).tolist()
                    count += 1
                    if count >= data_buf_len:
                        count = 0
                    state_now_id = average_probability.index(max(average_probability))

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

            #　認識結果をUDPで送信（受け取る側がpython2なのでprotocol=2を指定する）
            send_len = sock4probability.sendto(pickle.dumps(average_probability, protocol=2), serv_address)

            unnecessary_score_list = is_unnecessary_obj_list.mean(axis=0).tolist()
            send_len = sock4clean.sendto(pickle.dumps(unnecessary_score_list, protocol=2), serv_address4clean)
            frame_count += 1
            if frame_count == time_window:
                frame_count = 0
        else:
            pass

        spin_rate.sleep()