#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mediapipeによるposeデータを配信する。
このスクリプトはpython2では実行できないため
python3で実行できるワークスペースにあるパッケージに移植して実行する
"""
import sys
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

sys.path.append('/home/kubotalab-hsr/.pyenv/versions/3.8.10/lib/python3.8/site-packages')
import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


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
            self._input_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            # self._input_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)
    
    def get_image(self):
        return  self._input_image


def main():
    
    rospy.init_node('pose_publisher')

    pub_p = rospy.Publisher('mp_pose_data', Float32MultiArray, queue_size=1)

    landmark_idx = list(range(0,32))
        
    try:
        sub_image = ImageSbscriber(topic_name="/camera/rgb/image_raw")
        spin_rate = rospy.Rate(10)

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            # UpdateGUI Window
            while not rospy.is_shutdown():
                
                image = sub_image.get_image()
                if image is not None:
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

                    landmark_positions = []
                    if results.pose_landmarks is not None:
                        landmark = results.pose_landmarks.landmark
                        for idx in landmark_idx:
                            landmark_positions.append([landmark[idx].x, landmark[idx].y, landmark[idx].visibility])

                        # print(landmark_positions[0])
                        landmark_positions = Float32MultiArray(data=np.array(landmark_positions).flatten().tolist())
                        pub_p.publish(landmark_positions)
                    else:
                        landmark_positions = Float32MultiArray(data=[0.0,0.0,0.0])
                        pub_p.publish(landmark_positions)


                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    cv2.imshow("Detection Image Window", image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

                    spin_rate.sleep()

    except rospy.ROSException as wait_for_msg_exception:
        rospy.logerr(wait_for_msg_exception)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




