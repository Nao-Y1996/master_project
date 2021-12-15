#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from cv_bridge import CvBridge, CvBridgeError
import rospy
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import csv
import os
import tf as TF

class pose_subscriber():
    def __init__(self, topic_name):
        self.pose_data = np.array([0.0, 0.0, 0.0])

        # Subscribe color image data
        self._image_sub = rospy.Subscriber(topic_name, Float32MultiArray, self._call_back)
        # Wait until connection
        rospy.wait_for_message(topic_name, Float32MultiArray, timeout=5.0)

    def _call_back(self, data):
        self.pose_data = np.reshape(data.data, (-1,3))
    
    def get_pose(self):
        return  self.pose_data


def main(face_type=None):

    if (face_type is None) or (face_type not in ['f', 'r', 'l']):
        print('select the type --> type : f, r, l')
        sys.exit()
    base_dir = os.path.dirname(__file__)+'/FacePoseEstimation/'
    with open(base_dir+'face_landmark_'+face_type+'.csv', 'w') as f:
        print('created csv for face landmark')
    with open(base_dir+'rotation_'+face_type+'.csv', 'w') as f:
        print('created csv for face landmark')

    mp_pose = pose_subscriber(topic_name='mp_pose_data')
    listener = TF.TransformListener()
    br = TF.TransformBroadcaster()

    face_landmark_idx = list(range(0,11))
    tf = 'ar_marker_1'
    pre_rot = (0,0,0,0)
    spin_rate = rospy.Rate(10)

    count = 0
    pre_pose_data = np.array([[0.0, 0.0, 0.0]])
    Normalized_face_landmark_position =  [0.0, 0.0, 0.0]
    pre_Normalized_face_landmark_position =  [0.0, 0.0, 0.0]

    while not rospy.is_shutdown():
        human_detecting = False
        
        pose_data = mp_pose.get_pose()

        if (pre_pose_data[0][0] != pose_data[0][0]):
            pass
        else:
            pose_data = np.array([[0.0, 0.0, 0.0]])
        pre_pose_data = pose_data

        if pose_data.shape == (32,3):
            human_detecting = True
        
        if human_detecting:
            center_x, center_y = (pose_data[1][0] + pose_data[4][0])/2 , (pose_data[1][1] + pose_data[4][1])/2
            face_landmark_positions = []
            for idx in face_landmark_idx:
                face_landmark_positions.append([pose_data[idx][0], pose_data[idx][1]])
            
            Normalized_face_landmark_position = face_landmark_positions - np.array([center_x, center_y])
            Normalized_face_landmark_position = Normalized_face_landmark_position.flatten().tolist()
        
        
        if pre_Normalized_face_landmark_position[0]!=Normalized_face_landmark_position[0]:
            if len(Normalized_face_landmark_position) == 22:
                try:
                    trans, rot = listener.lookupTransform('/camera_rgb_optical_frame', tf, rospy.Time(0))
                    
                    # 向きを修正
                    rot = TF.transformations.euler_from_quaternion((rot[0],rot[1],rot[2],rot[3]))
                    if face_type == 'r':
                        rot = (rot[0]-1.57, rot[1], rot[2]) # 右頬
                    if face_type == 'f':
                        rot = (rot[0], rot[1]+1.57, rot[2]+1.57) # 正面
                    if face_type == 'l':
                        rot = (rot[0]+1.57, rot[1], rot[2]+3.14) # 左頬
                    
                    rot = TF.transformations.quaternion_from_euler(rot[0],rot[1],rot[2])

                    br.sendTransform(trans, rot, rospy.Time.now(), tf+'_ref',  '/camera_rgb_optical_frame')
                    
                    if (True in (abs(np.array(rot) - pre_rot) > 0.1)):
                        pass
                    else:
                        count += 1
                        print(count)
                        
                        with open(base_dir+'face_landmark_'+face_type+'.csv', 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(Normalized_face_landmark_position)
                            
                        with open(base_dir+'rotation_'+face_type+'.csv', 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(rot)
                            
                    pre_rot = rot.copy()
                except (TF.LookupException, TF.ExtrapolationException):
                    pass
        
        pre_Normalized_face_landmark_position = Normalized_face_landmark_position
    spin_rate.sleep()

if __name__ == '__main__':
    rospy.init_node('pose_sub')
    face_type = sys.argv[1]
    main(face_type)
