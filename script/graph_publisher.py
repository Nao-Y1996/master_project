#!/usr/bin/python
# -*- coding: utf-8 -*-
import enum
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import Float32MultiArray
import tf as TF
import os
import csv
from datetime import datetime
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import sys
from yolo_object_detector import GetYoloObjectInfo
import math
import pickle
import socket
from matplotlib import pyplot as plt
import pyautogui as pag
# import sqlite3
import pandas as pd
import time

rospy.init_node('listen', anonymous=True)
br = TF.TransformBroadcaster()
listener = TF.TransformListener()

class JointStateSbscriber(object):

    def __init__(self):
        self.topick_name = "/hsrb/joint_states"
        self.joint_positions = None

        # Subscribe color image data
        self._state_sub = rospy.Subscriber(self.topick_name, JointState, self._callback)
        # Wait until connection
        rospy.wait_for_message(self.topick_name, JointState, timeout=10.0)

    def _callback(self, data):
        self.joint_positions = data.position
    
    def get_head_position(self):
        return  self.joint_positions[9], self.joint_positions[10]


class FacePoseSubscriber():
    def __init__(self):
        self.face_x = None
        self.face_y = None
        self.roll ,self.pitch, self.yaw  = None,None,None
        self.face_sub = rospy.Subscriber("/head_pose_data", Float32MultiArray, self.callback)

    def callback(self, data):
        self.face_x, self.face_y = data.data[0], data.data[1]
        self.roll ,self.pitch, self.yaw = data.data[2], data.data[3], data.data[4]

    def get_face_center(self):
        try:
            return int(self.face_x), int(self.face_y)
        except TypeError:
            return self.face_x, self.face_y

    def get_rpy(self):
        return self.roll ,self.pitch, self.yaw 


class TF_Publisher():
# class TF_Publisher(JointStateSbscriber):
    def __init__(self, exe_type):
        if exe_type == 'xtion':  # for Xtion
            topic_name = "/camera/depth_registered/points"
            self.reference_tf = '/camera_depth_frame'
            
        elif exe_type == 'sense': # for realsense
            topic_name = "/camera/depth/color/points"
            self.reference_tf = '/camera_depth_frame'

        elif exe_type == 'hsr_sim': # for HSR simulator
            topic_name = "/hsrb/head_rgbd_sensor/depth_registered/points"
            self.reference_tf = 'head_rgbd_sensor_link'
            
        elif exe_type == 'hsr': # for HSR
            # super(TF_Publisher,self).__init__()
            topic_name = '/hsrb/head_rgbd_sensor/depth_registered/rectified_points'
            self.reference_tf = 'head_rgbd_sensor_link'
            # self.initial_head_pos = self.get_head_position()
        else:
            print('TF_Publisherクラスの初期化に失敗しました')
            sys.exit()
        self.exe_type = exe_type
        self.pc_sub = rospy.Subscriber(topic_name, PointCloud2, self.get_pc)
        rospy.wait_for_message(topic_name, PointCloud2, timeout=20.0)
        self.pc_data = None

    def get_pc(self, data):
        self.pc_data = data

    def create_object_TF(self, object_name, x, y, roll=None, pitch=None, yaw=None, create=True):
        if self.pc_data is not None:
            pc_list = list(pc2.read_points(self.pc_data,skip_nans=True,
                                                field_names=('x', 'y', 'z'),
                                                uvs=[(x, y)]))
            if len(pc_list) != 0:
                x,y,z = pc_list[0]

                if roll is not None  and self.exe_type == 'xtion':
                    rot = (math.radians(roll), math.radians(pitch-180), math.radians(yaw))
                    rot = TF.transformations.quaternion_from_euler(rot[0],rot[1],rot[2])
                elif roll is not None  and self.exe_type == 'hsr':
                    # pan, tilt = list(self.get_head_position()) - np.array(self.initial_head_pos)
                    # print(pan)
                    try:
                        trans, _ = listener.lookupTransform(self.reference_tf, 'face', rospy.Time(0))
                        r = np.linalg.norm([trans[0],trans[2]])
                        fai = np.arccos(trans[2]/r)
                        print(fai, np.degrees(fai))
                    except TF.LookupException:
                        pass
                    
                    rot = (math.radians(pitch), math.radians(-yaw-270), math.radians(roll))
                    rot = TF.transformations.quaternion_from_euler(rot[0],rot[1],rot[2])
                else:
                    rot = [1.0, 0.0, 0.0, 0.0]

                if create and self.exe_type == 'xtion':
                    br.sendTransform([z,-x,-y], rot, rospy.Time.now(), '/'+object_name, self.reference_tf)
                if create and self.exe_type == 'hsr':
                    br.sendTransform([x,y,z], rot, rospy.Time.now(), '/'+object_name, self.reference_tf)
                return x,y,z
            else:
                return None, None, None
        else:
            return None, None, None
        

class MediapipePoseSubscriber():
    def __init__(self):
        self.pose = np.array([0.0]*32, dtype=float)
        self.face_pose = np.array([[0.0]*2]*11, dtype=float)
        self.face_pose_visibility = np.array([0.0]*11, dtype=float)
        self.face_sub = rospy.Subscriber("/mp_pose_data", Float32MultiArray, self.callback)
        rospy.wait_for_message("/mp_pose_data", Float32MultiArray, timeout=10.0)

    def callback(self, data):
        self.pose = np.reshape(data.data,(-1,3))

    def get_face_center(self):
        self.face_pose  = np.mean(self.pose[0:11,0:2], axis=0)
        self.face_pose_visibility = self.pose[0:11,2]
        return self.face_pose, self.face_pose_visibility


# "face","bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
# "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", 
# "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
# "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"

MARKER_2_OBJECT ={}
OBJECT_NAME_2_ID ={}

obj_4_real = ["face","bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
               "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
               "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
               "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
obj_4_marker = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake" ]
marker_list = []
for i in range(1,len(obj_4_marker)+1):
    marker_list.append('ar_marker/'+str(700+i))

for marker, obj_name in zip(marker_list, obj_4_marker):
    MARKER_2_OBJECT[marker] = obj_name

objct_list = obj_4_real+obj_4_marker
for i, name in enumerate(objct_list):
    OBJECT_NAME_2_ID[name]=i


if __name__ == '__main__':
    args = sys.argv
    user_id = None
    user_id = int(args[1])
    rospy.set_param("/user_id", user_id)

    # 保存用ディレクトリの設定
    save_dir = rospy.get_param("/base_dir") +"/user_"+str(user_id)
    rospy.set_param("/save_dir", save_dir)
    image_dir = save_dir+'/images/'
    position_dir = save_dir+'/position_data/'
    try:
        os.makedirs(image_dir)
        os.makedirs(position_dir)
    except OSError:
        print('directory exist')

    db_file = save_dir+"/state.csv"
    is_known_user = os.path.isfile(db_file)

    if not is_known_user:
        #状態データベース(csv)を新規作成
        with open(db_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['state'])
        print('created csv file for state database')

        # 認識結果の保存用ファイルの作成
        probability_file_path = position_dir + '/porobability.csv'
        with open(probability_file_path, 'w') as f:
            print('created csv file for porobability')
    else:
        pass

    # 実行モードが１のときサーバーに接続(分類結果を受け取る)
    # if exe_mode == 1:
    #     print('connecting to server ...')
    #     # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     client.connect((data_server_ip, 12345)) #これでサーバーに接続します
    #     print('Successfuly connected to server')

    # data_server_ip = rospy.get_param('server_IP')
    # print(data_server_ip)
    # # UDPのためのソケットを作成する
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # クラスのインスタンス化
    # face_sub = FacePoseSubscriber()
    # joint_state_sub = JointStateSbscriber()
    pose_sub = MediapipePoseSubscriber()
    yolo_info = GetYoloObjectInfo()
    tf_pub = TF_Publisher(exe_type='hsr')

    # グラフデータ配信用のパブリッシャー
    graph_data_pub = rospy.Publisher('graph_data', Float32MultiArray, queue_size=10)

    spin_rate=rospy.Rate(100)
    count = 0
    pre_face_x = 0
    while not rospy.is_shutdown():
        
        # time.sleep(1)

        # pag.screenshot(image_dir+str(count)+'.jpg')
        face_exist = False
        obj_positions =[]

        # --------face pose---------
        '''
        x, y = face_sub.get_face_center()
        roll, pitch, yaw = face_sub.get_rpy()
        if (x is not None and y is not None) and (x!=0 and y!=0):
            face_x, face_y, face_z = tf_pub.create_object_TF('face', x, y, roll, pitch, yaw,create=True)
            if face_x is not None:
                print('face')
                obj_positions.append([OBJECT_NAME_2_ID['face'], face_x, face_y, face_z])
        '''
        face_center, visibility = pose_sub.get_face_center()
        if (visibility>0.5).all():
            try:
                face_x, face_y, face_z = tf_pub.create_object_TF('face', int(face_center[0]*640), int(face_center[1]*480), create=True)
                obj_positions.append([OBJECT_NAME_2_ID['face'], face_x, face_y, face_z])
                # print('face')
                face_exist = True
            except:
                import traceback
                traceback.print_exc()

        if face_exist:
            # -------- object detection ---------
            objects_info = yolo_info.get_objects() #"/darknet_ros/detection_image"
            if len(objects_info) > 0:
                detect_obj_list = []
                for obj in objects_info:
                    name = obj[0]
                    if (name != 'person') and (name in obj_4_real):
                        x, y = obj[1], obj[2]
                        obj_x, obj_y, obj_z = tf_pub.create_object_TF(name, x, y, create=True)
                        if obj_x is not None:
                            obj_positions.append( [OBJECT_NAME_2_ID[name] ,obj_x, obj_y, obj_z] )
                            # print(name)
                            pass
            # -------- marker detection ---------
            for marker in marker_list:
                exist_marker = False
                try:
                    _, _ = listener.lookupTransform('/map', marker, rospy.Time(0))
                    exist_marker = True
                except (TF.LookupException, TF.ExtrapolationException):
                    exist_marker = False
                    pass
                except:
                    exist_marker = False
                    import traceback
                    traceback.print_exc()
                    pass
                if exist_marker:
                    trans, rot = listener.lookupTransform(tf_pub.reference_tf, marker, rospy.Time(0))
                    obj_x, obj_y, obj_z = trans
                    obj_name = MARKER_2_OBJECT[marker]
                    # print(obj_name)
                    obj_positions.append( [OBJECT_NAME_2_ID[obj_name] ,obj_x, obj_y, obj_z] )
                    
                    # trans, rot = listener.lookupTransform(marker, marker, rospy.Time(0))
                    # br.sendTransform(trans, rot, rospy.Time.now(), obj_name,  marker)
                    # br.sendTransform(trans, rot, rospy.Time.now(), obj_name,  tf_pub.reference_tf)
        
        print(np.array(obj_positions)[:,0])
        graph_data = np.array(obj_positions).reshape(1,-1)[0].tolist()
        publish_data = Float32MultiArray(data=graph_data)
        graph_data_pub.publish(publish_data)

        obj_num = len(graph_data)/4
        robot_mode = rospy.get_param("/robot_mode")
        if robot_mode == 'graph_collecting' and (obj_num >= 2) and (face_x != pre_face_x):
            data_save_path = rospy.get_param("/data_save_path")
            with open(data_save_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(graph_data)
                pre_face_x = face_x

        

            # if exe_mode == 1:
            # if robot_mode == 'state_recognition':
            #     print(graph_data)
            #     data = pickle.dumps(graph_data)
                # client.send(data) #データを送信
                # send_len = sock.sendto(data, (data_server_ip,12345)) # UDPで送る
                # received_data = client.recv(1024) # データを受信
                # rx_meesage, addr = sock.recvfrom(1024)
                # print(rx_meesage.decode(encoding='utf-8'))
                # probability = np.array(pickle.loads(received_data))
                # with open(probability_file_path, 'a') as f:
                #     writer = csv.writer(f)
                #     writer.writerow(probability)
                # height = np.round(probability, decimals=5)*100
                # left = [1, 2, 3, 4 ]
                # plt.bar(left, height)
                # plt.ylim(0, 100)
                # plt.pause(0.001)
                # plt.cla()
            count += 1
        # if count >1000:
        #     break
        # spin_rate.sleep()
        print('------------------------------------------------>')