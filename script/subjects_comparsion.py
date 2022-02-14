#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from graph_tools import graph_utilitys
import matplotlib.pyplot as plt
from classificator_nnconv import classificator
import csv
import pandas as pd
import itertools

def show_probability_graph(labels, probability, count=None, is_save=False, save_dir=None, prob_by=None):
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    width = 0.35
    rects = ax.bar(x, probability, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.ylim(0, 1)
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.draw()  # 描画する。
    # plt.pause(0.01)  # 0.01 秒ストップする。
    # plt.cla()
    if is_save:
        fig.savefig(save_dir+'/'+str(count)+'_prob_by_'+prob_by+'.png')
    plt.close()



if __name__ == '__main__':

    # 各ユーザーのモデルでそれぞれのデータを推定する
    model_owner_list = ['kusakari', 'tou', 'ozawa']
    model_type = '/ideal_augmented_batch_nnconv.pt'
    target_csv_type = 'row_augmented_pattern'

    # ユーザー二人のデータで学習したモデルでそれぞれを推定する場合
    # model_owner_list =['comb-kusakari_ozawa', 'comb-kusakari_tou', 'comb-tou_ozawa']
    # model_type = '/ideal_batch_nnconv.pt'
    # target_csv_type = 'row_pattern'

    analyze_permutation = list(itertools.permutations(model_owner_list, 2))

    for user1, user2 in analyze_permutation:

        model_owner = user1
        target_user = user2
        print(model_owner, '--->', target_user)

        model_user_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/"+model_owner
        target_user_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/"+target_user

        save_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/user_comparsion/" + model_owner + '2' + target_user +'/'
        try:
            os.makedirs(save_dir+'images')
        except OSError:
            print('directory exist')
        
        # 認識モデルの設定
        model_owner_model_path = model_user_dir+model_type
        model_owner_cf = classificator(model=model_owner_model_path)
        target_model_path = target_user_dir+model_type
        target_cf = classificator(model=target_model_path)

        # ツールのインスタンス化
        ft_path = os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin'
        graph_utils = graph_utilitys(fasttext_model=ft_path)

        # target_userのデータを読み込む
        data_dir = target_user_dir+ '/position_data'
        csv_path_dict = {0:data_dir+'/'+target_csv_type+'_0.csv',1:data_dir+'/'+target_csv_type+'_1.csv',2:data_dir+'/'+target_csv_type+'_2.csv'}

        # 認識の確率表示のグラフ設定
        labels = ['working', 'eating', 'reading']

        # 分析結果のためのデータフレーム
        columun_names= ['data_id', 'true_label', 'is_result_same', 'result_by_model_owner', 'result_by_target', 'prob_by_model_owner','prob_by_target']
        df = pd.DataFrame(columns=columun_names,)
        analyze_data = [None]*len(columun_names)

        data_count = 0
        for true_label, csv_file_name in csv_path_dict.items():

            with open(csv_file_name) as f:
                csv_file = csv.reader(f)
                # dataを1ずつ読み込む
                for i, _row in enumerate(csv_file):
                    data = []
                    if '' in _row:
                        continue
                    for j, v in enumerate(_row):
                        data.append(float(v))

                    # dataをグラフ形式に変換
                    data_id = int(data[0])
                    position_data = graph_utils.removeDataId(data)
                    graph, node_names = graph_utils.positionData2graph(position_data, 10000, include_names=True)
                    position_data = np.reshape(position_data, (-1,4))
                    
                    if graph is not None:
                        # graph_utils.visualize_graph(graph, node_labels=node_names, save_graph_name=None, show_graph=True) # 状態グラフの表示
                        
                        # 状態認識
                        prob_by_model_owner = model_owner_cf.classificate(graph)
                        prob_by_target = target_cf.classificate(graph)

                        # 認識確率の表示
                        # show_probability_graph(labels, np.round(prob_by_model_owner, decimals=4).tolist(),is_save=True, count=data_id, save_dir=save_dir+'images', prob_by='model_owner')
                        # show_probability_graph(labels, np.round(prob_by_target, decimals=4).tolist(), is_save=True, count=data_id, save_dir=save_dir+'images',prob_by='target')
                        
                        # 認識結果
                        result_by_model_owner = prob_by_model_owner.index(max(prob_by_model_owner))
                        result_by_target = prob_by_target.index(max(prob_by_target))

                        # 認識結果が一致するか
                        is_result_same = result_by_model_owner==result_by_target

                        analyze_data = [data_id, true_label, is_result_same, result_by_model_owner, result_by_target, prob_by_model_owner,prob_by_target]
                        
                        df.loc[data_count] = analyze_data
                        data_count += 1
                            
        df.to_csv(save_dir+'analyzed.csv')
