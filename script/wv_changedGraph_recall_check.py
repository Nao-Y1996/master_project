import pandas as pd
import numpy as np
import csv
import os
from graph_converter import graph_utilitys
from classificator_gcn import classificator
import fasttext
import matplotlib.pyplot as plt

base_dir = os.path.abspath('')+'/experiment_data/SI'
csv_path_list = {0:base_dir+'/position_data/work.csv',
                 1:base_dir+'/position_data/meal_and_working_tools.csv',
                 2:base_dir+'/position_data/meal_while_working.csv',
                 3:base_dir+'/position_data/meal.csv'
                 }

obj_name_change_patterns = [
                            {'sandwich':'toast'},#1
                            {'sandwich':'book'},#2
                            {'orange':'grape'},#3
                            {'orange':'book'},#4
                            {'banana':'apple'},#5
                            {'banana':'book'},#6
                            {'donut':'cookie'},#7
                            {'donut':'book'},#8
                            {'sandwich':'toast','orange':'grape'},#9
                            {'sandwich':'book','orange':'chair'},#10
                            {'sandwich':'toast','orange':'grape','banana':'apple'},#11
                            {'sandwich':'book','orange':'chair','banana':'t-shirt'},#12
                            {'sandwich':'toast','orange':'grape','banana':'apple','donut':'cookie'},#13
                            {'sandwich':'book','orange':'chair','banana':'t-shirt','donut':'clock'},#14
                            {'laptop':'tvmonitor'},#15
                            {'laptop':'book'},#16
                            {'mouse':'cell phone'},#17
                            {'mouse':'book'},#18
                            {'keyboard':'iPad'},#19
                            {'keyboard':'book'},#20
                            {'laptop':'tvmonitor','mouse':'cell phone'},#21
                            {'laptop':'book','mouse':'chair'},#22
                            {'laptop':'tvmonitor','mouse':'cell phone','keyboard':'iPad'},#23
                            {'laptop':'book','mouse':'chair','keyboard':'t-shirt'}#24
                            ]

cf = classificator(model='SI_gcn-w300-30cm.pt')
model_path =  os.path.abspath('') +'/w2v_model/cc.en.300.bin'
ft = fasttext.load_model(model_path)
df = pd.DataFrame()
related_graph_ids = []
for i, pattern in enumerate(obj_name_change_patterns):
    save_dir = base_dir+'/RecallCheck-wvChangedPattern/pattern_'+str(i)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # インスタンスを作り直す（ID_2_OBJECT_NAMEをリセット）
    graph_util = graph_utilitys(fasttext_model=ft)
    # if i>0: # 認識物体の名前を変更
    graph_util.changeID_2_OBJECT_NAME(pattern)
    # データセットを作成
    datasets, obj_names_sets = graph_util.csv2graphDataset(csv_path_list, include_names=True)

    probabilitys = []
    related_graph_count = 0 # 物体名の変更が影響があるグラフの数
    crrect_num = 0
    is_related_graph_list = []
    for count,(graph, obj_names) in enumerate(zip(datasets, obj_names_sets)):
        is_related_graph = False
        # 物体名の変更が影響があるグラフの数
        for old_name in pattern.values():
            if old_name in obj_names:
                related_graph_count += 1
                is_related_graph = True
                break
        is_related_graph_list.append(is_related_graph)
        probability = cf.classificate(graph)
        result = np.argmax(probability)
        if is_related_graph and (result == int(graph.y)):
            crrect_num += 1
        if is_related_graph and (result != int(graph.y)):
            file_name = str(count)+'_'+str(result)+'->'+str(int(graph.y))+'.png'
            graph_util.visualize_graph(graph, node_labels=obj_names, save_graph_name=save_dir+'/'+file_name, show_graph=False)
        probabilitys.append(probability)
    print('パターン'+str(i+1)+' 正解率：\n' , crrect_num, '/', related_graph_count, ' = ', end='')
    try:
        print(crrect_num/related_graph_count)
    except:
        print('')
        pass
#     df['pattern_'+str(i)] = p_results
#     df['pattern_'+str(i)+'_is_related'] = is_related_graph_list
# df.to_csv( base_dir+'/RecallCheck-wvChangedPattern/to_csv_out.csv')
