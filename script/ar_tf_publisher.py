#!/usr/bin/python3
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
# from yolo_object_detector import GetYoloObjectInfo
# from graph_converter import converData2graph

import pickle
import socket
from matplotlib import pyplot as plt
# import pyautogui as pag
    
        
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

OBJECT_NAME_2_ID = {"robot":0,"bottle":1, "wine glass":2, "cup":3, "fork":4, "knife":5, "spoon":6, "bowl":7, "banana":8, "apple":9, "sandwich":10, 
                    "orange":11, "broccoli":12, "carrot":13, "hot dog":14, "pizza":15, "donut":16, "cake":17,"chair":18, "sofa":19, "pottedplant":20, 
                    "bed":21, "diningtable":22, "toilet":23,"tvmonitor":24, "laptop":25, "mouse":26, "remote":27, "keyboard":28, "cell phone":29, "microwave":30, 
                    "oven":31, "toaster":32, "sink":33, "refrigerator":34, "book":35, "clock":36, "vase":37, "scissors":38, "teddy bear":39, "hair drier":40, "toothbrush":41}

OBJECT_ID_2_NAME = {0: 'robot', 1: 'bottle', 2: 'wine glass', 3: 'cup', 4: 'fork', 5: 'knife', 6: 'spoon', 7: 'bowl', 8: 'banana', 9: 'apple',10: 'sandwich', 
                    11: 'orange', 12: 'broccoli', 13: 'carrot', 14: 'hot dog', 15: 'pizza', 16: 'donut', 17: 'cake', 18: 'chair',19: 'sofa', 20: 'pottedplant', 
                    21: 'bed', 22: 'diningtable', 23: 'toilet', 24: 'tvmonitor', 25: 'laptop', 26: 'mouse', 27: 'remote',28: 'keyboard', 29: 'cell phone', 30: 'microwave', 
                    31: 'oven', 32: 'toaster', 33: 'sink', 34: 'refrigerator', 35: 'book', 36: 'clock',37: 'vase', 38: 'scissors', 39: 'teddy bear', 40: 'hair drier', 41: 'toothbrush'}

TFs = ['ar_marker_0', 'ar_marker_1', 'ar_marker_2', 'ar_marker_3', 'ar_marker_4', 'ar_marker_5', 'ar_marker_6', 'ar_marker_7', 'ar_marker_8', 'ar_marker_9', 'ar_marker_10',
        'ar_marker_11', 'ar_marker_12', 'ar_marker_13', 'ar_marker_14', 'ar_marker_15', 'ar_marker_16', 'ar_marker_17', 'ar_marker_18', 'ar_marker_19', 'ar_marker_20',
        'ar_marker_21', 'ar_marker_22', 'ar_marker_23', 'ar_marker_24', 'ar_marker_25', 'ar_marker_26', 'ar_marker_27', 'ar_marker_28', 'ar_marker_29', 'ar_marker_30',
        'ar_marker_31', 'ar_marker_32', 'ar_marker_33', 'ar_marker_34', 'ar_marker_35', 'ar_marker_36', 'ar_marker_37', 'ar_marker_38', 'ar_marker_39', 'ar_marker_40', 'ar_marker_41']
# TFs = ['ar_marker_5', 'ar_marker_1']




if __name__ == '__main__':
    rospy.init_node('listen', anonymous=True)
    args = sys.argv
    exe_mode, file_name = None, None
    try:
        exe_mode = int(args[1])
        file_name = args[2]
    except:
        pass

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
        client.connect(('192.168.2.105', 50010)) #これでサーバーに接続します
        print('Successfuly connected to server')
        probability_file_path = position_dir + file_name + '_porobability.csv'
        with open(probability_file_path, 'w') as f:
            print('created csv file for porobability')


    spin_rate=rospy.Rate(100)
    count = 0
    listener = TF.TransformListener()
    br = TF.TransformBroadcaster()

    while not rospy.is_shutdown():

        # pag.screenshot(image_dir+str(count)+'.jpg')

        # --------face detection by mediapipe pose---------
        # print("face ---- ",face_sub.face_x, face_sub.face_y)
        # im = detectionimage.get_image()
        # cv2.imshow("image", im)
        # if face_sub.face_x is not None:
        #     face_x, face_y, face_z = tf_pub.create_object_TF('face', face_sub.face_x, face_sub.face_y, create=False)
        #     if face_x is not None:
        #         obj_positions.append([0, face_x, face_y, face_z])


        # AR markerの座標を取得
        obj_positions =[]
        for i, tf in enumerate(TFs):
            obj_id = None
            exist_obj = False
            if i == 0:
                obj_positions.append( [OBJECT_NAME_2_ID['robot'] ,0.0, 0.0, 0.0] ) #  camera(robot)をグラフノードに追加
                continue
            try:
                trans, rot = listener.lookupTransform('/map', tf, rospy.Time(0))
                exist_obj = True
            except (TF.LookupException, TF.ExtrapolationException):
                exist_obj = False
                pass
            except:
                exist_obj = False
                import traceback
                traceback.print_exc()
                pass
            # print(exist_obj)
            if exist_obj:
                trans, rot = listener.lookupTransform(tf, tf, rospy.Time(5))
                obj_id = i
                obj_name = OBJECT_ID_2_NAME[obj_id]
                br.sendTransform(trans, rot, rospy.Time.now(), obj_name,  tf)
                print(obj_name)


            
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