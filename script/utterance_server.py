#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import sys

import rospy
import os
from datetime import datetime
import pandas as pd
import csv

# ロボット機能を使うための準備
from hsrb_interface import Robot
robot = Robot()
tts = robot.try_get('default_tts')
whole_body = robot.try_get('whole_body')

whole_body.move_to_joint_positions({'head_pan_joint': -0.9, 
                                     'head_tilt_joint': -0.3,
                                     'arm_flex_joint': -2.6,
                                     'wrist_flex_joint':-0.5,
                                     'arm_lift_joint':0.6,
                                     'arm_roll_joint':0.0})
python_version = sys.version_info[0]

def exist_state_check(csv_file, state_name):
    df = pd.read_csv(csv_file)
    state_indexs = list(df[df['state']==state_name].index)
    if len(state_indexs) == 0:
        return False
    elif len(state_indexs) == 1:
        return True
    else:
        return None

def get_state_index(csv_file, state_name):
    df = pd.read_csv(csv_file)
    state_indexs = list(df[df['state']==state_name].index)
    if len(state_indexs) == 0:
        state_index = len(df)
    elif len(state_indexs) == 1:
        state_index = state_indexs[0]
    else:
        state_index = None
    return state_index

def get_stateName(csv_file, state_id):
    df = pd.read_csv(csv_file)
    name = df.iat[state_id, 0]
    return name

def add_new_state(csv_file, state_name):
    exist_state = exist_state_check(csv_file, state_name)
    if not exist_state:
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([state_name])
        print'新しい状態名を追加しました --->', state_name
    else:
        print('すでに状態名はあるので追加はスキップしました')


# 通信の設定
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
IP_ADDRESS = s.getsockname()[0]
M_SIZE = 1024
port = 8890
locaddr = (IP_ADDRESS, port)
print('utterance server : IP address = ', IP_ADDRESS, '  port = ', port)

# ①ソケットを作成する
sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
print('Successfully created socket!')
# ②使用するIPアドレスとポート番号を指定
sock.bind(locaddr)

if __name__ == "__main__":
    # user_name = input('enter user name')
    args = sys.argv
    user_name = None
    user_name = args[1]
    
    
    rospy.set_param("/user_name", user_name)
    rospy.set_param("/robot_mode", "nomal")
    rospy.set_param("/is_clean_mode", 0)

    # 保存用ディレクトリの設定
    user_dir = os.path.dirname(__file__)+'/experiment_data/'+user_name
    rospy.set_param("/user_dir", user_dir)

    while not rospy.is_shutdown():
        state_name = None
        state_index = None
        # rospy.set_param("/image_save_path", image_save_path)
        robot_mode = rospy.get_param("/robot_mode")
        try :
        # ③Clientからのmessageの受付開始
            message, cli_addr = sock.recvfrom(M_SIZE)
            # python2では（）つけるとコンソールの表示が崩れる
            if python_version == 2:
                print message
            if python_version == 3:
                message = message.decode(encoding='utf-8')
                print(message)
            
            if 'モードの確認' in message:
                robot_mode = rospy.get_param("/robot_mode")
                is_clean_mode = rospy.get_param("/is_clean_mode")
                if robot_mode == "state_recognition":
                    tts.say('現在、認識モードです。')
                elif robot_mode == "nomal":
                    tts.say('現在、通常モードです。')
                elif robot_mode == "waite_state_name":
                    tts.say('現在、記録の準備中です。 今、何をしているか教えてもらえたら記録を開始できます。')
                elif robot_mode == "graph_collecting":
                    tts.say('現在、記録中です。')
                else:
                    tts.say('モードが不明です。')
                if is_clean_mode:
                    tts.say('片付け機能が働いています。')

            elif '終了' in message:
                if (robot_mode=='graph_collecting'):
                    save_dir = rospy.get_param("/save_dir")
                    user_dir = rospy.get_param("/user_dir")
                    db_file = user_dir+"/state.csv"

                    state_index = rospy.get_param("/state_index")
                    state_name = get_stateName(db_file, state_index)
                    

                    image_save_path = save_dir+'/images/'
                    rospy.set_param("/image_save_path", image_save_path)
                    rospy.set_param("/robot_mode", "nomal")

                    tts.say(state_name + 'の記録は完了です。')
                else:
                    pass
            
            elif '認識モード' in message:
                rospy.set_param("/robot_mode", "state_recognition")
                rospy.set_param("/is_clean_mode", 0)
                tts.say('はい、認識機能をオンにします')
            
            elif '通常モード' in message:
                rospy.set_param("/robot_mode", "nomal")
                rospy.set_param("/is_clean_mode", 0)
                tts.say('はい、モードを切り替えました。')

            elif '記録して' in message:
                rospy.set_param("/robot_mode", "waite_state_name")
                tts.say('はい、今何をしていますか？')

            elif '片付け' in message:
                rospy.set_param("/robot_mode", "state_recognition")
                rospy.set_param("/is_clean_mode", 1)
                tts.say('はい、不要なものを探します。')

            elif 'ありがとう' in message:
                rospy.set_param("/is_clean_mode", 0)

            elif 'はい' in message:
                pass
            elif '今' in message:
                pass

            else :
                robot_mode = rospy.get_param("/robot_mode")
                if (robot_mode=='waite_state_name'):
                    # state_name = input('状態名を入力してください')
                    state_name = message
                    tts.say(state_name + '、を記録します。')
                    
                    save_dir = rospy.get_param("/save_dir")
                    user_dir = rospy.get_param("/user_dir")
                    db_file = user_dir+"/state.csv"

                    state_index = get_state_index(db_file, state_name)
                    add_new_state(db_file, state_name)

                    rospy.set_param("/state_index", state_index)

                    # 収集するデータを保存するファイルを指定
                    image_save_path = save_dir+'/images/pattern_'+str(state_index)+'/'
                    data_save_path = save_dir+'/position_data/pattern_'+str(state_index)+'.csv'
                    print(data_save_path + ' にデータを保存します')
                    rospy.set_param("/data_save_path", data_save_path)
                    rospy.set_param("/image_save_path", image_save_path)
                    
                    # データ収集モードに切り替え
                    rospy.set_param("/robot_mode", "graph_collecting")
                    print(state_name +' のデータを収集します')

            

        except KeyboardInterrupt:
            print ('\n . . .\n')
            sock.close()
            break
        except socket.timeout:
            pass
        except:
            import traceback
            traceback.print_exc()
        
