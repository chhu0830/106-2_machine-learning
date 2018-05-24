#!/usr/bin/env python3
# -*- coding: utf-8 -*-

train_image = [[float(x) for x in line.strip().split(",")] for line in open("./data/X_train.csv")]
train_label = [float(line.strip()) for line in open("./data/T_train.csv")]

for label, image in zip(train_label, train_image):
    print(int(label), end=" ")
    for i, pixel in enumerate(image):
        if float(pixel) != 0.0:
            print(i, pixel, sep=":", end=" ")
    print()
