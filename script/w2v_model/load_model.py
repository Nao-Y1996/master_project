#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
 
import fasttext as ft
import fasttext.util

model_path = os.path.dirname(__file__))+'/cc.en.300.bin'
if not os.path.exists(model_path):
    fasttext.util.download_model('en', if_exists='ignore')
model_path = os.path.dirname(__file__))+'/cc.en.50.bin'
if not os.path.exists(model_path):
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, 50)
    ft.save_model(model_path)

ft = fasttext.load_model(model_path)