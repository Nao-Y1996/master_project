import pandas as pd
import numpy as np
import csv
import os
from graph_converter import graph_utilitys
from classificator_nnconv import classificator
import fasttext
import matplotlib.pyplot as plt

base_dir = os.path.abspath('')+'/experiment_data/yamada/position_data/'
csv_path_list = {0:base_dir+'ideal_augmented_pattern_0.csv',
                 1:base_dir+'ideal_augmented_pattern_1.csv',
                 2:base_dir+'ideal_augmented_pattern_2.csv',
                 }
# ==============物体名の入れ替えパターンを生成=========================
# 組み合わせの総パターンを得る（７個の物体をへんこうする時は１２７通り）
import itertools
obj_name_change_patterns = {0:{'sandwich':'bagel'},
                            1:{'soup':'chowder'},
                            2:{'salad':'coleslaw'},
                            3:{'book':'comic'},
                            4:{'laptop':'tablet'},
                            5:{'keyboard':'trackpad'},
                            6:{'mouse':'trackpad'}}
all_pattern = []
count = 1
for i in range(1,8):
    # print(f'{i}個の物体名を入れ替えるパターン')
    comb  = list(itertools.combinations(list(range(7)), i))
    for key_num_list in comb:
        pattern = {}
        for num in list(key_num_list):
            pattern.update(obj_name_change_patterns[num])
        # print(f'変更パターン{count} : {pattern}')
        count += 1
        all_pattern.append(pattern)
print(len(all_pattern))
# ===================================================================

cf = classificator(model='./experiment_data/yamada/ideal_augmented_batch_nnconv.pt')
model_path =  os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin'
# ft = fasttext.load_model(model_path)
df = pd.DataFrame()
related_graph_ids = []
length = None
with open(base_dir+'RecallCheck-wvChangedPattern.csv', 'r') as f:
    reader = csv.reader(f)
    length = len(list(reader))
for i, pattern in enumerate(all_pattern):
    print(i, pattern)
    if i < length:
        continue
    change_num =len(pattern.items())
    # f = open(base_dir+'RecallCheck-wvChangedPattern.txt', 'a')
    # f.write(str(pattern).replace('{','').replace('}','').replace(':',' -->').replace("'", "")+'\n')
    # f.close()
    # save_dir = base_dir+'/RecallCheck-wvChangedPattern/pattern_'+str(i)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # インスタンスを作り直す（ID_2_OBJECT_NAMEをリセット）
    graph_util = graph_utilitys(fasttext_model=model_path)
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
        # if is_related_graph and (result != int(graph.y)):
        #     file_name = str(count)+'_'+str(result)+'->'+str(int(graph.y))+'.png'
        #     graph_util.visualize_graph(graph, node_labels=obj_names, save_graph_name=save_dir+'/'+file_name, show_graph=False)
        probabilitys.append(probability)
    try:
        ans = str(crrect_num/related_graph_count)
    except:
        ans = ''
        pass
    # print('パターン'+str(i+1)+' 正解率\n' , crrect_num, '/', related_graph_count, ' = ', end='')
    write_data = [i, change_num, ans, crrect_num, related_graph_count, str(pattern).replace('{','').replace('}','').replace(': ','-->').replace("'", "").replace(", ", ":")]
    with open(base_dir+'RecallCheck-wvChangedPattern.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(write_data)
        
    # f = open(base_dir+'RecallCheck-wvChangedPattern.txt', 'a')
    # f.write('パターン'+str(i+1)+' 正解率\n')
    # f.write(str(crrect_num) + ' / ' + str(related_graph_count)+' = ' + ans + '\n')
    # f.close()
    
    # if i>2:
    #     break
#     df['pattern_'+str(i)] = p_results
#     df['pattern_'+str(i)+'_is_related'] = is_related_graph_list
# df.to_csv( base_dir+'/RecallCheck-wvChangedPattern/to_csv_out.csv')
