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
print('IP address = ',IP_ADDRESS)
M_SIZE = 1024
host = IP_ADDRESS
port = 8890
locaddr = (host, port)

# ①ソケットを作成する
sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
print('Successfully created socket!')

# ②使用するIPアドレスとポート番号を指定
sock.bind(locaddr)

if __name__ == "__main__":
    # rospy.init_node('main_controller', anonymous=True)
    
    rospy.set_param("/robot_mode", "nomal")

    # 保存用ディレクトリの設定
    base_dir = os.path.dirname(__file__)+'/experiment_data/'+str(datetime.now()).split(' ')[0]
    rospy.set_param("/base_dir", base_dir)

    while not rospy.is_shutdown():
        state_name = None
        state_index = None
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
                print(robot_mode)

            elif '終了' in message:
                save_dir = rospy.get_param("/save_dir")
                db_file = save_dir+"/state.csv"

                state_index = rospy.get_param("/state_index")
                state_name = get_stateName(db_file, state_index)
                print(state_name + ' のデータ収集を終了します')
                rospy.set_param("/robot_mode", "finish_graph_collecting")
            
            elif '認識モード' in message:
                rospy.set_param("/robot_mode", "state_recognition")
            
            elif '通常モード' in message:
                rospy.set_param("/robot_mode", "nomal")

            elif '覚えて' in message:
                rospy.set_param("/robot_mode", "waite_state_name")
                tts.say('はい、今何をしていますか？')

            elif 'はい' in message:
                pass
            elif '今' in message:
                pass

            else :
                robot_mode = rospy.get_param("/robot_mode")
                if (robot_mode=='waite_state_name'):
                    # state_name = input('状態名を入力してください')
                    state_name = message
                    tts.say(state_name + '、を記録します')
                    
                    save_dir = rospy.get_param("/save_dir")
                    db_file = save_dir+"/state.csv"

                    state_index = get_state_index(db_file, state_name)
                    add_new_state(db_file, state_name)

                    rospy.set_param("/state_index", state_index)

                    # 収集するデータを保存するファイルを指定
                    data_save_path = save_dir+'/position_data/pattern_'+str(state_index)+'.csv'
                    print(data_save_path + ' にデータを保存します')
                    rospy.set_param("/data_save_path", data_save_path)
                    
                    # データ収集モードに切り替え
                    rospy.set_param("/robot_mode", "graph_collecting")
                    print(state_name +' のデータを収集します')

            

        except KeyboardInterrupt:
            print ('\n . . .\n')
            sock.close()
            break
        except:
            import traceback
            traceback.print_exc()
        
