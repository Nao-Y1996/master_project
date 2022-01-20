#!/usr/bin/env python
# Copyright (C) 2016 Toyota Motor Corporation
"""Flesh Color Detection Sample"""
import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import Image


class ImageSbscriber(object):

    def __init__(self, topic_name):
        # topic_name = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        self._bridge = CvBridge()
        self._input_image = None

        # Subscribe color image data
        self._image_sub = rospy.Subscriber(
            topic_name, Image, self._color_image_cb)
        # Wait until connection
        rospy.wait_for_message(topic_name, Image, timeout=5.0)

    def _color_image_cb(self, data):
        try:
            self._input_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)
    
    def get_image(self):
        return  self._input_image


def main():
    rospy.init_node('image_sub')
    try:
        sub_image = ImageSbscriber(topic_name="/darknet_ros/detection_image")
        # sub_image = ImageSbscriber(topic_name="/camera/rgb/image_raw")
        spin_rate = rospy.Rate(30)

        # UpdateGUI Window
        while not rospy.is_shutdown():
            dst_image = sub_image.get_image()
            cv2.imshow("Detection Image Window", dst_image)
            cv2.waitKey(3)
            spin_rate.sleep()

    except rospy.ROSException as wait_for_msg_exception:
        rospy.logerr(wait_for_msg_exception)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()