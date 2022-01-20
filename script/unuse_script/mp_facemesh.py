#!/usr/bin/env python3
import sys
# print(sys.version_info[0])
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import tf as TF
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

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


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def main():
    rospy.init_node('image_sub')
    sub_image = ImageSbscriber(topic_name="/camera/rgb/image_raw")
    spin_rate = rospy.Rate(30)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while not rospy.is_shutdown():
                
            image = sub_image.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print('-------------------------')
            print(results.multi_face_landmarks[0].landmark)
            print('-------------------------')
            for landmark in results.multi_face_landmarks[0].landmark:
                
            
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

if __name__ == '__main__':
    main()
