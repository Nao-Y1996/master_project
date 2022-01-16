#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np

from graph_converter import graph_utilitys
import matplotlib.pyplot as plt
from classificator_gcn import classificator
import traceback
graph_utils = graph_utilitys(fasttext_model='cc.en.300.bin')


class DataSbscriber(object):

    def __init__(self, topic_name):
        self.data = None
        self._image_sub = rospy.Subscriber(topic_name, Float32MultiArray, self.callback)
        rospy.wait_for_message(topic_name, Float32MultiArray, timeout=5.0)

    def callback(self, data):
        self.data = data.data
    
    def get_data(self):
        return  self.data
    

if __name__ == '__main__':

    rospy.init_node('master_model_nnconv', anonymous=True)
    spin_rate=rospy.Rate(10)
    print("start-----------------------------------")
    cf = classificator(model='SI_gcn-w300-30cm.pt')

    data_sub = DataSbscriber(topic_name='graph_data')

    while not rospy.is_shutdown():
        robot_mode = rospy.get_param("/robot_mode")
        clean_mode = rospy.get_param("/is_clean_mode")
        
        probability_pub = rospy.Publisher('probability', Float32MultiArray, queue_size=1)

        # データを受け取る
        data = data_sub.get_data()
        # グラフに変換
        position_data = graph_utils.removeDataId(data)
        graph, node_names = graph_utils.positionData2graph(position_data, 10000, include_names=True)
        print(node_names)
        
        # ノードを１つ取り除いたパターンのグラフを取得
        dummy_graph_lsit, removed_obj_data_list = graph_utils.convertData2dummygraphs(data)

        # グラフの表示
        if graph is not None:
            # graph_utils.visualize_graph(graph, node_labels=node_names,
            #                 save_graph_name=None, show_graph=True)
            
            # 状態認識
            if robot_mode == 'state_recognition':
                probability = cf.classificate(graph)
                print(probability)

                # 不要な物体（ノード）の特定
                if clean_mode:
                    for dummy_graph, removed_obj_data in zip(dummy_graph_lsit, removed_obj_data_list):
                        dummy_probability = cf.classificate(dummy_graph)
                        # あるノードを取り除いた時の認識結果ともとの認識結果が一致するか
                        if dummy_probability.index(dummy_probability.max()) == probability.index(probability.max()):
                            # あるノードを取り除いた時の認識結果の確率が上昇するか
                            if dummy_probability.max() > probability.max():
                                removed_obj_id = removed_obj_data[0]
                                unnecessary_obj = graph_utils.ID_2_OBJECT_NAME[int(removed_obj_id)]
                                print('=======不要ノード========')
                                print(unnecessary_obj)
                                print('=======================')
                            else:
                                print('確率は上昇しませんでした')
                        else:
                            print('認識結果が一致していません')
                else:
                    pass

                try:
                    publish_data = Float32MultiArray(data=probability)
                    probability_pub.publish(publish_data)
                except rospy.exceptions.ROSSerializationException:
                    # 認識モードから通常モードへの切替時に
                    # rospy.exceptions.ROSSerializationException: field data[] must be float type
                    # のエラーが出る　のでそれ用
                    continue
                except:
                    traceback.print_exc()

        spin_rate.sleep()