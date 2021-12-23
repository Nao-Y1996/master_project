#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField,Image
import cv2
import tf as TF
import os
import time
import csv
from datetime import datetime
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import sys
from master_project.msg import FaceCenter
# from face_detector import FaceDetection
from sub_image import ImageSbscriber
from yolo_object_detector import GetYoloObjectInfo
# from graph_converter import converData2graph

import pickle
import socket
from matplotlib import pyplot as plt
import pyautogui as pag

br = TF.TransformBroadcaster()
    
class TF_Publisher():
    def __init__(self, exe_type):
        if exe_type == 'xtion':  # for Xtion
            topic_name = "/camera/depth_registered/points"
            self.reference_tf = '/camera_depth_frame'
            
        elif exe_type == 'sense': # for realsense
            topic_name = "/camera/depth/color/points"
            self.reference_tf = 'head_rgbd_sensor_link'

        elif exe_type == 'hsr_sim': # for HSR simulator
            topic_name = "/hsrb/head_rgbd_sensor/depth_registered/points"
            self.reference_tf = 'head_rgbd_sensor_link'
            
        elif exe_type == 'hsr': # for HSR
            topic_name = '/hsrb/head_rgbd_sensor/depth_registered/rectified_points'
            self.reference_tf = 'head_rgbd_sensor_link'
        else:
            print('TF_Publisherクラスの初期化に失敗しました')
            sys.exit()
        self.exe_type = exe_type
        self.pc_sub = rospy.Subscriber(topic_name, PointCloud2, self.get_pc)
        rospy.wait_for_message(topic_name, PointCloud2, timeout=10.0)
        self.pc_data = None

    def get_pc(self, data):
        self.pc_data = data

    def create_object_TF(self, object_name,x,y,create=True):
        if self.pc_data is not None:
            pc_list = list(pc2.read_points(self.pc_data,skip_nans=True,
                                                field_names=('x', 'y', 'z'),
                                                uvs=[(x, y)]))
            if len(pc_list) != 0:
                x,y,z = pc_list[0]
                if create and self.exe_type == 'xtion':
                    br.sendTransform([z,-x,-y], [1.0, 0.0, 0.0, 0.0], rospy.Time.now(), '/'+object_name, self.reference_tf)
                if create and self.exe_type == 'hsr':
                    br.sendTransform([x,y,z], [1.0, 0.0, 0.0, 0.0], rospy.Time.now(), '/'+object_name, self.reference_tf)
                return x,y,z
            else:
                return None, None, None
        else:
            return None, None, None
        
class FaceSubscriber():
    def __init__(self):
        self.face_x = None
        self.face_y = None
        face_sub = rospy.Subscriber("/face_center", FaceCenter, self.get_face_center)
    def get_face_center(self, data):
        self.face_x, self.face_y = data.face_x, data.face_y
        # print("face ---- ",self.face_x, self.face_y)



OBJECT_LIST = ["bottle", "wine glass", "cup", "fork", "knife", 
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", 
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", 
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# OBJECT_LIST = ["bottle", "wine glass", "cup", "fork", "knife", 
#     "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
#     "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
#     "chair", "sofa", "pottedplant", "bed", "toilet", 
#     "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", 
#     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# OBJECT_NAME_2_ID = {"bottle":0, "wine glass":1, "cup":2, "fork":3, "knife":4, 
#     "spoon":5, "bowl":6, "banana":7, "apple":8, "sandwich":9, "orange":10, 
#     "broccoli":11, "carrot":12, "hot dog":13, "pizza":14, "donut":15, "cake":16,
#     "chair":17, "sofa":18, "pottedplant":19, "bed":20, "diningtable":21, "toilet":22,
#     "tvmonitor":23, "laptop":24, "mouse":25, "remote":26, "keyboard":27, "cell phone":28, 
#     "microwave":29, "oven":30, "toaster":31, "sink":32, "refrigerator":33, "book":34, "clock":35, "vase":36, "scissors":37, "teddy bear":38, "hair drier":39, "toothbrush":40}

# OBJECT_NAME_2_ID = {"face":0,"bottle":1, "wine glass":2, "cup":3, "fork":4, "knife":5, 
#     "spoon":6, "bowl":7, "banana":8, "apple":9, "sandwich":10, "orange":11, 
#     "broccoli":12, "carrot":13, "hot dog":14, "pizza":15, "donut":16, "cake":17,
#     "chair":18, "sofa":119, "pottedplant":20, "bed":21, "diningtable":22, "toilet":23,
#     "tvmonitor":24, "laptop":25, "mouse":26, "remote":27, "keyboard":28, "cell phone":29, 
#     "microwave":30, "oven":31, "toaster":32, "sink":33, "refrigerator":34, "book":35, "clock":36, "vase":37, "scissors":38, "teddy bear":39, "hair drier":40, "toothbrush":41}

OBJECT_NAME_2_ID = {"robot":0,"bottle":1, "wine glass":2, "cup":3, "fork":4, "knife":5, 
    "spoon":6, "bowl":7, "banana":8, "apple":9, "sandwich":10, "orange":11, 
    "broccoli":12, "carrot":13, "hot dog":14, "pizza":15, "donut":16, "cake":17,
    "chair":18, "sofa":119, "pottedplant":20, "bed":21, "diningtable":22, "toilet":23,
    "tvmonitor":24, "laptop":25, "mouse":26, "remote":27, "keyboard":28, "cell phone":29, 
    "microwave":30, "oven":31, "toaster":32, "sink":33, "refrigerator":34, "book":35, "clock":36, "vase":37, "scissors":38, "teddy bear":39, "hair drier":40, "toothbrush":41}


if __name__ == '__main__':
    rospy.init_node('listen', anonymous=True)
    args = sys.argv
    exe_mode, file_name = None, None
    try:
        exe_mode = int(args[1])
        file_name = args[2]
    except:
        pass
    
    server_ip = rospy.get_param('server_IP')

    # 保存用ディレクトリの設定
    base_dir = os.path.dirname(__file__)+'/experiment_data/'
    new_dir_path = str(datetime.now())[0:16].replace(' ', '-').replace(':', '-')
    save_dir = base_dir + new_dir_path
    # 各フレームにおけるobject positionとimageの保存用先
    try:
        os.makedirs(save_dir+'/images/')
        os.makedirs(save_dir+'/position_data/')
    except OSError:
        print('file exist')
    image_dir = save_dir+'/images/'
    position_dir = save_dir+'/position_data/'
    # データを保存するファイル名の設定
    if file_name is None:
        file_name = 'test'
    position_file_path = position_dir + file_name + '.csv'
    with open(position_file_path, 'w') as f:
        print('created csv file for position')
    # 実行モードが１のときサーバーに接続(分類結果を受け取る)
    if exe_mode == 1:
        print('connecting to server ...')
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #オブジェクトの作成をします
        client.connect((server_ip, 12345)) #これでサーバーに接続します
        print('Successfuly connected to server')
        probability_file_path = position_dir + file_name + '_porobability.csv'
        with open(probability_file_path, 'w') as f:
            print('created csv file for porobability')

    # detectionimage = ImageSbscriber(topic_name="/darknet_ros/detection_image")
    # face_sub = FaceSubscriber()
    yolo_info = GetYoloObjectInfo()
    tf_pub = TF_Publisher(exe_type='hsr')

    spin_rate=rospy.Rate(100)
    count = 0
    while not rospy.is_shutdown():

        pag.screenshot(image_dir+str(count)+'.jpg')

        obj_positions =[]

        # --------face detection by mediapipe pose---------
        # print("face ---- ",face_sub.face_x, face_sub.face_y)
        # im = detectionimage.get_image()
        # cv2.imshow("image", im)
        # if face_sub.face_x is not None:
        #     face_x, face_y, face_z = tf_pub.create_object_TF('face', face_sub.face_x, face_sub.face_y, create=False)
        #     if face_x is not None:
        #         obj_positions.append([0, face_x, face_y, face_z])

        # camera(robot)をグラフノードに追加
        obj_positions.append( [OBJECT_NAME_2_ID['robot'] ,0.0, 0.0, 0.0] )

        # --------object detection by YOLO---------
        objects_info = yolo_info.get_objects()#"/darknet_ros/detection_image"
        time.sleep(0.1)
        if len(objects_info) > 0:
            detect_obj_list = []
            for obj in objects_info:
                name = obj[0]
                if (name != 'person') and (name in OBJECT_LIST):
                    x, y = obj[1], obj[2]
                    obj_x, obj_y, obj_z = tf_pub.create_object_TF(name, x, y, create=True)
                    
                    if obj_x is not None:
                        # objects_position[OBJECT_NAME_2_ID[name]] = [obj_x, obj_y, obj_z]
                        obj_positions.append( [OBJECT_NAME_2_ID[name] ,obj_x, obj_y, obj_z] )
                        # print(name, obj_x, obj_y, obj_z)
                        print(name)
            
            graph_info = np.array(obj_positions).reshape(1,-1)[0].tolist()
            with open(position_file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(graph_info)

            if exe_mode == 1:
                data = pickle.dumps(graph_info)
                client.send(data) #データを送信
                received_data = client.recv(1024) # データを受信
                probability = np.array(pickle.loads(received_data))
                with open(probability_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(probability)
                height = np.round(probability, decimals=5)*100
                left = [1, 2, 3, 4 ]
                plt.bar(left, height)
                plt.ylim(0, 100)
                plt.pause(0.001)
                plt.cla()
            count += 1
            # print(count)
        # if count >1000:
        #     break
        spin_rate.sleep()
        # time.sleep(0.1)
        print('------------------------------------------------>')