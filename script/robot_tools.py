#!/usr/bin/python
# -*- coding: utf-8 -*-
class RobotPartner():
    def __init__(self, exe_type, hsr_robot):
        self.exe_type = exe_type
        self.tts = None
        self.whole_body = None
        if self.exe_type == 'hsr':
            self.whole_body = hsr_robot.try_get('whole_body')
            self.tts = hsr_robot.try_get('default_tts')
    def init_pose(self):
        self.whole_body.move_to_joint_positions({'head_pan_joint': 0.9, 
                                 'head_tilt_joint': -0.3,
                                 'arm_flex_joint': -2.6,
                                 'wrist_flex_joint':-0.5,
                                 'arm_lift_joint':0.6,
                                 'arm_roll_joint':0.0})
    def say(self, content):
        if self.exe_type=='hsr':
            self.tts.say(content)
        else:
            print('Robot : ' + content)