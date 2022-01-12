#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import json

# "face","bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
# "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", 
# "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
# "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"

conf_dir = os.path.dirname(__file__)+'/obj_conf/'

MARKER_2_OBJECT ={}
OBJECT_NAME_2_ID ={}
ID_2_OBJECT_NAME = {}

obj_4_real = ["face","bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
               "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
               "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
               "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

obj_4_marker = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake" ]
marker_list = []

for i in range(1,len(obj_4_marker)+1):
    marker_list.append('ar_marker/'+str(700+i))

for marker, obj_name in zip(marker_list, obj_4_marker):
    MARKER_2_OBJECT[marker] = obj_name

objct_list = obj_4_real+obj_4_marker
for i, name in enumerate(objct_list):
    OBJECT_NAME_2_ID[name] = i
    ID_2_OBJECT_NAME[i] = name

with open(conf_dir+'ID_2_OBJECT_NAME.json', 'w') as f:
    json.dump(ID_2_OBJECT_NAME, f)

with open(conf_dir+'OBJECT_NAME_2_ID.json', 'w') as f:
    json.dump(OBJECT_NAME_2_ID, f)

with open(conf_dir+'MARKER_2_OBJECT.json', 'w') as f:
    json.dump(MARKER_2_OBJECT, f)


'''
id2obj = {}
with open(conf_dir+'ID_2_OBJECT_NAME.json') as f:
    _id2obj = json.load(f)
    for k, v in _id2obj.items():
        id2obj[int(k)] = v
print(id2obj)

print('----------------------------------')

with open(conf_dir+'OBJECT_NAME_2_ID.json') as f:
    obj2id = json.load(f)
print(obj2id)

print('----------------------------------')

with open(conf_dir+'MARKER_2_OBJECT.json') as f:
    marker2obj = json.load(f)
print(marker2obj)
'''