#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import sys

import rospy
# import rosparam

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
                print (message)
            if python_version == 3:
                message = message.decode(encoding='utf-8')
                print(message)
            
            if 'モードの確認' in message:
                robot_mode = rospy.get_param("/robot_mode")
                print(robot_mode)

            if '覚えて' in message:
                robot_mode = rospy.get_param("/robot_mode")
                if robot_mode != 'graph_collecting':
                    pattern_count = rospy.get_param("/state_pattern_count")
                    # データ収集モードに切り替え
                    rospy.set_param("/robot_mode", "graph_collecting")
                    print(str(pattern_count)+'パターン目のデータを収集します')

            if '終了' in message:
                # データ収集モードをしゅうりょうする
                rospy.set_param("/robot_mode", "finish_graph_collecting")

                pattern_count = rospy.get_param("/state_pattern_count")
                print(str(pattern_count)+'パターン目のデータ収集を終了')
                rospy.set_param("/state_pattern_count", pattern_count+1)
            
            if '認識モード' in message:
                # データ収集モードをしゅうりょうする
                rospy.set_param("/robot_mode", "state_recognition")

        except KeyboardInterrupt:
            print ('\n . . .\n')
            sock.close()
            break
        except:
            import traceback
            traceback.print_exc()
        
