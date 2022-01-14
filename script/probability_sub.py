#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


class ProbabilitySbscriber(object):

    def __init__(self, topic_name):
        self.probability = None
        self._probability_sub = rospy.Subscriber(topic_name, Float32MultiArray, self.callback)
        rospy.wait_for_message(topic_name, Float32MultiArray, timeout=5.0)

    def callback(self, data):
        self.probability = data.data
    
    def get_probability(self):
        return  self.probability


if __name__ == '__main__':

    rospy.init_node('graph_subscriber', anonymous=True)
    spin_rate=rospy.Rate(10)

    probability_sub = ProbabilitySbscriber(topic_name='probability')

    state_names = []
    save_dir = rospy.get_param("/save_dir")
    # with open(save_dir +'/state.csv','r') as f:
    #     csvreader = csv.reader(f)
    #     for row in csvreader:
    #         state_names.append(row[0])
    read_data = pd.read_csv(save_dir +'/state.csv',encoding="utf-8")
    state_names = read_data['state'].tolist()

    while not rospy.is_shutdown():
        robot_mode = rospy.get_param("/robot_mode")


        # グラフデータを受け取る
        probability = probability_sub.get_probability()


        print(probability)
        print   state_names[0].encode('utf-8'), state_names[1].encode('utf-8')
        height = np.round(probability, decimals=5)*100
        left = [1, 2, 3, 4 ]
        plt.bar(left, height)
        plt.ylim(0, 100)
        plt.pause(0.001)
        plt.cla()

        spin_rate.sleep()