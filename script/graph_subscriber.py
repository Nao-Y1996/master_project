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


class GraphSbscriber(object):

    def __init__(self, topic_name):
        self.graph_data = None
        self.graph = None
        self.names = None
        self._image_sub = rospy.Subscriber(topic_name, Float32MultiArray, self.callback)
        rospy.wait_for_message(topic_name, Float32MultiArray, timeout=5.0)

    def callback(self, data):
        self.graph, self.names = graph_utils.convertData2graph(data.data, 10000, include_names=True)
    
    def get_grapf(self):
        return  self.graph
    
    def get_node_names(self):
        return  self.names

if __name__ == '__main__':

    rospy.init_node('graph_subscriber', anonymous=True)
    spin_rate=rospy.Rate(10)
    print("start-----------------------------------")
    cf = classificator(model='SI_gcn-w300-30cm.pt')

    graph_sub = GraphSbscriber(topic_name='graph_data')

    while not rospy.is_shutdown():
        robot_mode = rospy.get_param("/robot_mode")

        probability_pub = rospy.Publisher('probability', Float32MultiArray, queue_size=1)

        # グラフデータを受け取る
        graph = graph_sub.get_grapf()
        node_names = graph_sub.get_node_names()
        print(node_names)

        # グラフの表示
        if graph is not None:
            # graph_utils.visualize_graph(graph, node_labels=node_names,
            #                 save_graph_name=None, show_graph=True)
            
            # 状態認識
            if robot_mode == 'state_recognition':
                probability = cf.classificate(graph)
                print(probability)

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