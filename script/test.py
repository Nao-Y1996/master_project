#!/usr/bin/python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import random

# a = [random.random(), random.random(), random.random(), random.random()]

for _ in range(1000):
    height = [random.random(), random.random(), random.random(), random.random()]
    left = [1, 2, 3, 4 ]
    plt.bar(left, height)
    plt.ylim(0,1.0)
    plt.pause(0.01)
    plt.cla()