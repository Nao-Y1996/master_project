#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import sys
import rospy
import os
import pandas as pd
import csv
from robot_tools import RobotPartner

PYTHON_VERSION = sys.version_info[0]

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
        print '新しい状態を追加しました --->', state_name
    else:
        # print('すでに状態名はあるので追加はスキップしました')
        pass


if __name__ == "__main__":

    # 音声認識結果を受け取るための通信の設定
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0] # get IP address of this PC
    port = 8890
    print('utterance server : IP address = '+str(IP_ADDRESS)+'  port = '+str(port))
    sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
    sock.bind((IP_ADDRESS, port))
    print('Successfully created socket!')

    # usernameと実行タイプの入力
    """
    実行タイプ
    hsr     : real robot
    hsr_sim : HSR simulator
    xtion   : xtion camera
    """
    if PYTHON_VERSION==2:
        user_name = raw_input('\nenter user name\n')
        exe_type = raw_input("select exe type hsr, hsr_sim, xtion\n")
    elif PYTHON_VERSION==3:
        user_name = input('\nenter user name\n')
        exe_type = input("select exe type hsr, hsr_sim, xtion\n")
    else:
        sys.exit("Unexpected python version is used")
    
    if exe_type!="hsr" and exe_type!="hsr_sim" and exe_type!="xtion":
        sys.exit("select correct exe type (hsr/hsr_sim/xtion)")
    elif exe_type=='hsr' or exe_type=='hsr_sim':
        # HSRロボット機能のimport
        from hsrb_interface import Robot
        hsr_robot = Robot()
        robotPartner = RobotPartner(exe_type=exe_type,hsr_robot=hsr_robot)
    elif exe_type=='xtion':
        robotPartner = RobotPartner(exe_type=exe_type,hsr_robot=None)
    else:
        pass
    
    rospy.set_param("/user_name", user_name)
    rospy.set_param("/robot_mode", "nomal")
    rospy.set_param("/is_clean_mode", 0)
    rospy.set_param("/exe_type", exe_type)

    # 保存用ディレクトリの設定
    user_dir = os.path.dirname(__file__)+'/experiment_data/'+user_name
    rospy.set_param("/user_dir", user_dir)

    print('-------- start --------\n')
    while not rospy.is_shutdown():
        state_name = None
        state_index = None
        robot_mode = rospy.get_param("/robot_mode")
        try :# ③Clientからのmessageの受付開始
        
            message, cli_addr = sock.recvfrom(1024)
            # python2では（）つけるとコンソールの表示が崩れる
            if PYTHON_VERSION == 2:
                print message
            if PYTHON_VERSION == 3:
                message = message.decode(encoding='utf-8')
                print(message)

            if ('はい' in message) or ('します'in message) or ('現在'in message) or ('今'in message) or ('完了'in message):
                pass
            
            elif 'モードの確認' == message:
                robot_mode = rospy.get_param("/robot_mode")
                is_clean_mode = rospy.get_param("/is_clean_mode")
                if robot_mode == "state_recognition":
                    robotPartner.say('現在、認識モードです。')
                elif robot_mode == "nomal":
                    robotPartner.say('現在、通常モードです。')
                elif robot_mode == "waite_state_name":
                    robotPartner.say('現在、記録の準備中です。 今、何をしているか教えてもらえたら記録を開始できます。')
                elif robot_mode == "graph_collecting":
                    robotPartner.say('現在、記録中です。')
                else:
                    robotPartner.say('モードが不明です。')
                if is_clean_mode:
                    robotPartner.say('片付け機能が働いています。')

            elif '終了' == message:
                if (robot_mode=='graph_collecting'):
                    save_dir = rospy.get_param("/save_dir")
                    user_dir = rospy.get_param("/user_dir")
                    db_file = user_dir+"/state.csv"

                    state_index = rospy.get_param("/state_index")
                    state_name = get_stateName(db_file, state_index)
                    
                    image_save_path = save_dir+'/images/'
                    rospy.set_param("/image_save_path", image_save_path)
                    rospy.set_param("/robot_mode", "nomal")
                    rospy.set_param("/cllecting_state_name", '')

                    robotPartner.say(state_name + 'の記録は完了です。')
                else:
                    pass
            
            elif '認識モード'== message:
                can_recognize = os.path.exists(user_dir+'/model_info.json')
                if can_recognize:
                    rospy.set_param("/robot_mode", "state_recognition")
                    rospy.set_param("/is_clean_mode", 0)
                    robotPartner.say('はい、認識機能をオンにします。')
                else:
                    robotPartner.say('利用可能な認識モデルがありません。学習後に認識モードが利用可能になります。')
            
            # elif 'モデルの学習を開始' in message:
            # ここで、自動でモデルの学習（python3で動く）ができるようになるのが理想
            # ubuntu20とROS noeticを使用すればシステム全体がpython3で動かせるので簡単に実現できるが、
            # 現状、ubuntu18（基本的にpython2）だと自動化は少し面倒
            
            elif '通常モード' == message:
                rospy.set_param("/robot_mode", "nomal")
                rospy.set_param("/is_clean_mode", 0)
                robotPartner.say('はい、通常機能に戻ります。')

            elif '片付け' == message:
                rospy.set_param("/robot_mode", "state_recognition")
                rospy.set_param("/is_clean_mode", 1)
                robotPartner.say('はい、不要なものを探します。')

            elif 'ありがとう' == message:
                rospy.set_param("/is_clean_mode", 0)
                robotPartner.say('はい、どういたしまして')

            elif '記録して' == message:
                rospy.set_param("/robot_mode", "waite_state_name")
                robotPartner.say('はい、今何をしていますか？')

            else :
                robot_mode = rospy.get_param("/robot_mode")
                if (robot_mode=='waite_state_name'):
                    state_name = message
                    robotPartner.say(state_name + '、を記録します。')
                    
                    save_dir = rospy.get_param("/save_dir")
                    user_dir = rospy.get_param("/user_dir")
                    db_file = user_dir+"/state.csv"

                    state_index = get_state_index(db_file, state_name)
                    add_new_state(db_file, state_name)

                    rospy.set_param("/state_index", state_index)

                    # 収集するデータを保存するファイルを指定
                    image_save_path = save_dir+'/images/pattern_'+str(state_index)+'/'
                    data_save_path = save_dir+'/position_data/raw_pattern_'+str(state_index)+'.csv'
                    print(data_save_path + ' にデータを保存します')
                    rospy.set_param("/data_save_path", data_save_path)
                    rospy.set_param("/image_save_path", image_save_path)
                    
                    # データ収集モードに切り替え
                    rospy.set_param("/robot_mode", "graph_collecting")
                    print(state_name +' のデータを収集します')
                    rospy.set_param("/collecting_state_name", state_name)
                elif (robot_mode=='finish_collecting'):
                    robotPartner.say('記録は完了です')
                    rospy.set_param("/robot_mode", "nomal")
                else:
                    pass

            

        except KeyboardInterrupt:
            print ('\n . . .\n')
            sock.close()
            break
        except socket.timeout:
            pass
        except:
            import traceback
            traceback.print_exc()
        
