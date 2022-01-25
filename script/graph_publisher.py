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
np.set_printoptions(precision=6, suppress=True)
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
import traceback
import json

# ロボット機能を使うための準備
from hsrb_interface import Robot
robot = Robot()
tts = robot.try_get('default_tts')
whole_body = robot.try_get('whole_body')

# rospy.init_node('listen', anonymous=True)
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

conf_dir = os.path.dirname(__file__)+'/obj_conf/'
MARKER_2_OBJECT ={}
OBJECT_NAME_2_ID ={}
ID_2_OBJECT_NAME = {}

obj_4_real = ["face", "tvmonitor", "laptop", "mouse", "keyboard", "book", "banana", "apple", "orange", "pizza","cup"]
obj_4_marker = ['toast', 'sandwich', 'cereal', 'scrambled egg', 'soup', 'salada', 'donut']
additional_obj = ['bottle']#,'chair']

marker_list = []
for i in range(1,len(obj_4_marker)+1):
    marker_list.append('ar_marker/'+str(700+i))

for marker, obj_name in zip(marker_list, obj_4_marker):
    MARKER_2_OBJECT[marker] = obj_name

for i, name in enumerate(obj_4_real + obj_4_marker + additional_obj):
    OBJECT_NAME_2_ID[name]=i
    ID_2_OBJECT_NAME[i] = name

with open(conf_dir+'ID_2_OBJECT_NAME.json', 'w') as f:
    json.dump(ID_2_OBJECT_NAME, f)

with open(conf_dir+'OBJECT_NAME_2_ID.json', 'w') as f:
    json.dump(OBJECT_NAME_2_ID, f)

with open(conf_dir+'MARKER_2_OBJECT.json', 'w') as f:
    json.dump(MARKER_2_OBJECT, f)



if __name__ == '__main__':
    user_name = rospy.get_param("/user_name")
    print
    print('=============================')
    print('current user is '+user_name)
    print('=============================')
    print
    rospy.sleep(3)
    # while True:
    #     y_n = input('Do you continue? (y/n)')
    #     if y_n == 'y':
    #         break
    #     elif y_n == 'n':
    #         sys.exit("finished program")
    #     else:
    #         pass
    user_dir = rospy.get_param("/user_dir")

    # 保存用ディレクトリの設定
    #time_now = str(datetime.now()).split(' ')
    save_dir = user_dir # + '/'+time_now[0] + '-' +  time_now[1].split('.')[0].replace(':', '-')
    rospy.set_param("/save_dir", save_dir)
    image_dir = save_dir+'/images/'
    rospy.set_param("/image_save_path", image_dir)
    position_dir = save_dir+'/position_data/'
    recognition_file_path = position_dir + '/recognition.csv'

    try:
        os.makedirs(image_dir)
        os.makedirs(position_dir)
    except OSError:
        print('directory exist')

    db_file = user_dir+"/state.csv"
    is_known_user = os.path.isfile(db_file)

    if not is_known_user:
        #状態データベース(csv)を新規作成
        with open(db_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['state'])
        print('created csv file for state database')

        # 認識結果の保存用ファイルの作成
        with open(recognition_file_path, 'w') as f:
            print('created csv file for recognition')
        
        for i in range(10):
            try:
                os.makedirs(image_dir+'pattern_'+str(i))
            except OSError:
                print('directory exist')
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

    # データ送信のためのソケットを作成する（UDP）
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serv_address = ('192.168.0.109', 12345)

    # グラフ収集：保存した数の送信用
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serv_address1 = ('192.168.0.109', 54321)
    # グラフ収集：検出物体名の送信用
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serv_address2 = ('192.168.0.109', 56789)

    # クラスのインスタンス化
    # face_sub = FacePoseSubscriber()
    # joint_state_sub = JointStateSbscriber()
    pose_sub = MediapipePoseSubscriber()
    yolo_info = GetYoloObjectInfo()
    tf_pub = TF_Publisher(exe_type='hsr')

    # グラフのパブリッシャー
    # graph_data_pub = rospy.Publisher('graph_data', Float32MultiArray, queue_size=1)

    spin_rate=rospy.Rate(10)
    count_saved = 0
    count_ideal_saved = 0
    recognition_count = 0
    pre_graph_data = None
    while not rospy.is_shutdown():

        robot_mode = rospy.get_param("/robot_mode")
        is_clean_mode = rospy.get_param("/is_clean_mode")

        if is_clean_mode or (robot_mode == 'state_recognition'):
            detectable_obj_lsit = obj_4_real + additional_obj
        else:
            detectable_obj_lsit = obj_4_real

        obj_moved = False
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
                traceback.print_exc()
        names = []
        if face_exist:
            # -------- object detection ---------
            objects_info = yolo_info.get_objects() #"/darknet_ros/detection_image"
            if len(objects_info) > 0:
                detect_obj_list = []
                for obj in objects_info:
                    name = obj[0]
                    if (name != 'person') and (name in detectable_obj_lsit):
                        x, y = obj[1], obj[2]
                        obj_x, obj_y, obj_z = tf_pub.create_object_TF(name, x, y, create=True)
                        if obj_x is not None:
                            obj_positions.append( [OBJECT_NAME_2_ID[name] ,obj_x, obj_y, obj_z] )
                            # print(name)
                            names.append(name)
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
                    traceback.print_exc()
                    pass
                if exist_marker:
                    trans, rot = listener.lookupTransform(tf_pub.reference_tf, marker, rospy.Time(0))
                    obj_x, obj_y, obj_z = trans
                    obj_name = MARKER_2_OBJECT[marker]
                    names.append(obj_name)
                    # print(obj_name)
                    obj_positions.append( [OBJECT_NAME_2_ID[obj_name] ,obj_x, obj_y, obj_z] )
                    
                    # trans, rot = listener.lookupTransform(marker, marker, rospy.Time(0))
                    # br.sendTransform(trans, rot, rospy.Time.now(), obj_name,  marker)
                    br.sendTransform(trans, rot, rospy.Time.now(), obj_name,  tf_pub.reference_tf)

        
        graph_data = np.array(obj_positions).reshape(1,-1)[0].tolist()

        # 先頭にdata_idを追加
        data_id = int(float(time.time())*100)
        graph_data.insert(0, float(data_id))

        try:
            graph_diff = np.array(pre_graph_data[1:]) - graph_data[1:]
            all_obj_moved = map(lambda k: abs(k)>0.02, graph_diff) # 前回保存したデータと比較して各オブジェクトが2cm以上移動しているかどうか
            if any(all_obj_moved):
                obj_moved = True
            else:
                obj_moved = False
        except (TypeError, ValueError):
            obj_moved = True
            pass            
        except:
            traceback.print_exc()


        #------------ データをUDPで送る ------------#
        send_len = sock.sendto(pickle.dumps(graph_data), serv_address)
        #----------------------------------------------#
        
        obj_num = (len(graph_data)-1)/4
        if robot_mode == 'graph_collecting':
            state_name = rospy.get_param("/cllecting_state_name")
            # ------------ データの送信 ------------#
            # グラフ収集：保存した数の送信用
            print(count_ideal_saved)
            send_len1 = sock1.sendto(pickle.dumps(count_ideal_saved), serv_address1)
            # グラフ収集：検出物体名の送信用
            send_len2 = sock2.sendto(pickle.dumps(names), serv_address2)
            #----------------------------------------------#
            # data_save_path = rospy.get_param("/data_save_path") 
            if face_exist:
                if (obj_num >= 2) and obj_moved:
                # if len(set(names))>=8 and (obj_num >= 2) and obj_moved: # yamada
                    # print(names)
                    # print state_name
                    data_save_path = rospy.get_param("/data_save_path") # row_pattern_n.csv
                    ideal_data_save_path = data_save_path.replace('row', 'ideal') # ideal_pattern_n.csv
                    if state_name == '読書'.decode('utf-8'):
                        can_save_IdealData = True if ('book' in names)else False
                    elif state_name == '仕事'.decode('utf-8'):
                        can_save_IdealData = True if (('laptop' in names) and \
                                                      ('tvmonitor' in names)) else False
                    elif state_name == '昼食'.decode('utf-8'):
                        can_save_IdealData = True if ( ('sandwich' in names) ) else False
                    else:
                        can_save_IdealData = False

                    if can_save_IdealData:
                        with open(ideal_data_save_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(graph_data)
                        pre_graph_data = graph_data
                        count_ideal_saved += 1
                    else:
                        pass
            
                    with open(data_save_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(graph_data)
                    pre_graph_data = graph_data
                    count_saved += 1
                else:
                    pass
                #　スクリーンショット
                image_save_path = rospy.get_param("/image_save_path")
                pag.screenshot(image_save_path+str(data_id)+'.jpg')
            else:
                pass
            print('保存したデータ数 : '+str(count_saved) + '  |  保存した理想データ数 : ' + str(count_ideal_saved))

            if count_ideal_saved >= 1000:
                save_dir = rospy.get_param("/save_dir")
                image_save_path = save_dir+'/images/'
                rospy.set_param("/image_save_path", image_save_path)
                rospy.set_param("/robot_mode", "nomal")
                rospy.set_param("/cllecting_state_name", '')
                tts.say('記録は完了です。')
        elif robot_mode == 'state_recognition':
            with open(recognition_file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(graph_data)
                #　スクリーンショット
                image_save_path = rospy.get_param("/image_save_path")
                pag.screenshot(image_save_path+str(data_id)+'.jpg')
                recognition_count += 1
            print(names)
            # if recognition_count >1000:
            #     tts.say('終了')
            #     rospy.set_param("/robot_mode", "nomal")

            
        else:
            count_saved = 0
            count_ideal_saved = 0
            recognition_count = 0
            print(names)
            

        spin_rate.sleep()
 
        
        # print('------------------------------------------------')
