#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import sys

import rospy
# import rosparam

# 通信の設定
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
IP_ADDRESS = s.getsockname()[0]
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
    rospy.init_node('main_controller', anonymous=True)
    
    rospy.set_param("/robot_mode", "nomal")
    rospy.set_param("/state_pattern_count", 1)

    while not rospy.is_shutdown():
        try :
        # ③Clientからのmessageの受付開始
            message, cli_addr = sock.recvfrom(M_SIZE)
            # python2では（）つけるとコンソールの表示が崩れる
            python_version = sys.version_info[0]
            if python_version == 2:
                print message, type(message)
            if python_version == 3:
                message = message.decode(encoding='utf-8')
                print(message, type(message))
            if '覚えて' in message:
                print('GNN学習データの収集を開始します')
                pattern_count = rospy.get_param("/state_pattern_count")
                rospy.set_param("/state_pattern_count", pattern_count+1)
                # データ収集モードに切り替え
                rospy.set_param("/robot_mode", "graph_collecting")

            if '終了' in message:
                print('GNN学習データの収集を開始します')
                # データ収集モードをしゅうりょうする
                rospy.set_param("/robot_mode", "finish_graph_collecting")
            
            if '認識モード' in message:
                print('状態認識を開始します')
                # データ収集モードをしゅうりょうする
                rospy.set_param("/robot_mode", "state_recognition")

        except KeyboardInterrupt:
            print ('\n . . .\n')
            sock.close()
            break
        except:
            import traceback
            traceback.print_exc()
        
