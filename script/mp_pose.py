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
            self._input_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)
    
    def get_image(self):
        return  self._input_image


def main(face_type=None):
    if (face_type is None) or (face_type not in ['f', 'r', 'l']):
        print('select the type --> type : f, r, l')
        sys.exit()
    rospy.init_node('image_sub')
    face_landmark_idx = list(range(0,11))
    
    listener = TF.TransformListener()
    br = TF.TransformBroadcaster()
    tf = 'ar_marker_1'
    pre_rot = (0,0,0,0)
    
    base_dir = os.path.dirname(__file__)+'/FacePoseEstimation/'
    
    with open(base_dir+'face_landmark_'+face_type+'.csv', 'w') as f:
        print('created csv for face landmark')
    with open(base_dir+'rotation_'+face_type+'.csv', 'w') as f:
        print('created csv for face landmark')
        
    try:
        # sub_image = ImageSbscriber(topic_name="/darknet_ros/detection_image")
        sub_image = ImageSbscriber(topic_name="/camera/rgb/image_raw")
        spin_rate = rospy.Rate(30)

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            # UpdateGUI Window
            count = 0
            while not rospy.is_shutdown():
                Normalized_face_landmark_position = None
                
                image = sub_image.get_image()
                

                # save face landmark
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                if results.pose_landmarks is not None:
                    landmark = results.pose_landmarks.landmark
                    center_x, center_y = (landmark[1].x + landmark[4].x)/2 , (landmark[1].y + landmark[4].y)/2
                    face_landmark_positions = []
                    for idx in face_landmark_idx:
                        face_landmark_positions.append([landmark[idx].x, landmark[idx].y])
                    
                    Normalized_face_landmark_position = face_landmark_positions - np.array([center_x, center_y])
                    Normalized_face_landmark_position = np.array(Normalized_face_landmark_position).flatten().tolist()
                
                

                if Normalized_face_landmark_position is not None:
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
                        
                        
                        
                

                # Draw the pose annotation on the image.
                # h, w, _ = np.shape(image)
                # cv2.circle(image, (int(w*center_x), int(h*center_y)), 15, (0, 0, 255), thickness=-1)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
    face_type = sys.argv[1]
    main(face_type)




# For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()