#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import sensor_msgs.point_cloud2
from roslib import message
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField,Image
import ros_numpy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf as TF
import os

# br = TF.TransformBroadcaster()
class FaceDetection():
    def __init__(self):
        self.cv_bridge = CvBridge()
        topic_name = "/camera/rgb/image_raw"
        self.im_sub = rospy.Subscriber(topic_name, Image, self.callback_image)
        rospy.wait_for_message(topic_name, Image, timeout=5.0)
        self.faces = None
        self.rgb_image = None
        self.cascade_path = os.path.dirname(rectified_pointsrectified_points
            __file__) + '/haarcascades/haarcascade_frontalface_alt2.xml'

    def callback_image(self, data):
        try:
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def detect_face_centers(self):
        cascade = cv2.CascadeClassifier(self.cascade_path)
        self.faces = list(cascade.detectMultiScale(
            self.rgb_image, scaleFactor=1.9, minNeighbors=2))
        face_centers = []
        for (x, y, w, h) in self.faces:
            face_center_x = x+w//2
            face_center_y = y+h//2
            # self.face_areas.append(w * h)
            face_centers.append([face_center_x, face_center_y])
            # cv2.rectangle(input_image, (x, y), (x+w, y+h),
            #               color=(0, 0, 225), thickness=3)
            # cv2.circle(input_image, (face_center_x, face_center_y), 10, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8, shift=0)
            # print(self.face_areas, self.face_centers)
        return face_centers, self.rgb_image

