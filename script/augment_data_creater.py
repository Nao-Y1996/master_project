#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import random
import csv
import os
import glob
import sys
import json


def augment(graph_utils, user_name, data_type):
    print('\n================== Start Data Augmentation ==================')

    # 拡張前のデータファイル(csv)を辞書に格納する
    origin_csv_path_dict = {}
    user_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/"+user_name
    files = glob.glob(user_dir + "/PositionData_4_Train/"+data_type+"_pattern*")
    files.sort()
    pattern_num = len(files)
    print('---- original data file ----')
    for i, file in enumerate(files):
        origin_csv_path_dict[i] = file
        print(i, file)
    
    # 各状態パターンごとの「時々使う物体」をリストに格納する
    sometimes_obj_list = []
    obj_combinations_file = os.path.dirname(__file__)+'/experiment_data/'+ user_name + '/obj_combinations.json'
    with open(obj_combinations_file) as f:
        _dict = json.load(f)
        for i in range(pattern_num):
            sometimes_obj_list.append(list(map(str,_dict['state_pattern'+str(i)]['sometimes'])))
    print('---- 「時々使う物体」 ----')
    for i, objs in enumerate(sometimes_obj_list):
        print(i, objs)
        if len(objs)==0:
            print(f'状態パターン{i}に「時々使う物体」の設定がありません')
            is_continue = input(f'このまま続けますか？ (y/n)（yを選択すると、状態パターン{i}の学習用データでは"augmented_raw/ideal_data"は"raw/ideal_data"と等しくなります。）')
            if is_continue=='y':
                pass
            elif is_continue=='n':
                sys.exit('終了しました。「時々使用する物体」「必ず使用する物体」の設定は"obj_combinations.json"でできます。')
            else:
                sys.exit('select y or n')
    # すべての状態において「時々使用する」物体が設定されていない時はデータの拡張ができないため、終了する
    if np.array(sometimes_obj_list, dtype=object).size==0:
        sys.exit("すべての状態において「時々使用する」物体が設定されていないため、でーたのかくちょうができません. obj_combinations.jsonで設定してください")
    

    # 「時々使用する物体」のidのリストを得る
    remove_obj_ids_list = []
    for remove_obj_names in sometimes_obj_list:
        remove_obj_ids = []
        for name in remove_obj_names:
            remove_obj_ids.append(graph_utils.OBJECT_NAME_2_ID[name])
        remove_obj_ids_list.append(remove_obj_ids)
    data_num_list = []
    for origin_csv, remove_obj_ids in zip(origin_csv_path_dict.values(), remove_obj_ids_list):
        augmented_csv = origin_csv.replace('pattern', 'augmented_pattern') # (data_type)_pattern.csv --> (data_type)_augmented_pattern.csv
        # データを拡張して保存　＆　データ数を返す
        data_num = graph_utils.CreateAugmentedCSVdata(origin_csv, augmented_csv, remove_obj_ids)
        data_num_list.append(data_num)

    # 拡張した学習データのデータ数を揃える（一番少ないものに合わせる。多いものはランダムに選択）
    augmented_csv_path_dict = {}
    min_data_num = min(data_num_list)
    print("データ数 : ",data_num_list, ' 最小 : ',min_data_num)
    print(f'データ数を{min_data_num}に揃えます')
    for k, v in origin_csv_path_dict.items():
        augmented_csv_path_dict[k] = v.replace('pattern', 'augmented_pattern')
    
    # データ数が揃えられた拡張データを保存し直す
    for file in augmented_csv_path_dict.values():
        with open(file) as f:
            csv_file = csv.reader(f)
            data = [[float(v) for v in row] for row in csv_file]
        data = random.sample(data, min_data_num)
        with open(file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    
    return augmented_csv_path_dict