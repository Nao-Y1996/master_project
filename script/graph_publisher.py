#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
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

    def create_object_TF(self, object_name, x, y, roll=None, pitch=None, yaw=None, create=True):
        if self.pc_data is not None:
            pc_list = list(pc2.read_points(self.pc_data,skip_nans=True,
                                                field_names=('x', 'y', 'z'),
                                                uvs=[(x, y)]))
            if len(pc_list) != 0:
                x,y,z = pc_list[0]

                if roll is not None:
                    rot = (math.radians(roll), math.radians(pitch-180), math.radians(yaw))
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

OBJECT_LIST = ["bottle", "wine glass", "cup", "fork", "knife", 
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", 
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", 
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

OBJECT_NAME_2_ID = {"face":0,"bottle":1, "wine glass":2, "cup":3, "fork":4, "knife":5, 
    "spoon":6, "bowl":7, "banana":8, "apple":9, "sandwich":10, "orange":11, 
    "broccoli":12, "carrot":13, "hot dog":14, "pizza":15, "donut":16, "cake":17,
    "chair":18, "sofa":119, "pottedplant":20, "bed":21, "diningtable":22, "toilet":23,
    "tvmonitor":24, "laptop":25, "mouse":26, "remote":27, "keyboard":28, "cell phone":29, 
    "microwave":30, "oven":31, "toaster":32, "sink":33, "refrigerator":34, "book":35, "clock":36, "vase":37, "scissors":38, "teddy bear":39, "hair drier":40, "toothbrush":41}


if __name__ == '__main__':
    rospy.init_node('listen', anonymous=True)

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

    # 10パターン分のデータを保存するファイルを作成
    for i in range(10):
        file_name = 'pattern_'+str(i)
        position_file_path = position_dir + file_name + '.csv'
        with open(position_file_path, 'w') as f:
            print('created csv file for position')

    # 認識結果の保存用ファイルの作成
    probability_file_path = position_dir + file_name + '_porobability.csv'
    with open(probability_file_path, 'w') as f:
        print('created csv file for porobability')

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
    face_sub = FacePoseSubscriber()
    yolo_info = GetYoloObjectInfo()
    tf_pub = TF_Publisher(exe_type='xtion')

    # グラフデータ配信用のパブリッシャー
    graph_data_pub = rospy.Publisher('graph_data', Float32MultiArray, queue_size=10)

    spin_rate=rospy.Rate(100)
    count = 0
    while not rospy.is_shutdown():

        # pag.screenshot(image_dir+str(count)+'.jpg')

        obj_positions =[]

        # --------face pose---------
        x, y = face_sub.get_face_center()
        roll, pitch, yaw = face_sub.get_rpy()
        # print("face ----> ",x, y)
        if (x is not None and y is not None) and (x!=0 and y!=0):
            face_x, face_y, face_z = tf_pub.create_object_TF('face', x, y, roll, pitch, yaw,create=True)
            if face_x is not None:
                print('face')
                obj_positions.append([OBJECT_NAME_2_ID['face'], face_x, face_y, face_z])

        # camera(robot)をグラフノードに追加
        # obj_positions.append( [OBJECT_NAME_2_ID['robot'] ,0.0, 0.0, 0.0] )

        # --------object detection by YOLO---------
        objects_info = yolo_info.get_objects()#"/darknet_ros/detection_image"
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
            
            graph_data = np.array(obj_positions).reshape(1,-1)[0].tolist()
            publish_data = Float32MultiArray(data=graph_data)
            graph_data_pub.publish(publish_data)

            robot_mode = rospy.get_param("/robot_mode")

            if robot_mode == 'graph_collecting':
                pattern = rospy.get_param("/state_pattern_count")
                file_name = 'pattern_'+str(pattern)
                position_file_path = position_dir + file_name + '.csv'
                with open(position_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(graph_data)

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
        spin_rate.sleep()
        print('------------------------------------------------>')