#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys

train_images = [[float(x) for x in line.strip().split(",")] for line in open("../data/X_train.csv")]
train_labels = [int(line.strip()) for line in open("../data/T_train.csv")]
test_images = [[float(x) for x in line.strip().split(",")] for line in open("../data/X_test.csv")]
test_labels = [int(line.strip()) for line in open("../data/T_test.csv")]

""" Precompute dot table """
def gen_table(imagesA, imagesB, fileName):
    dot_table = [[0 for _ in range(len(imagesB))] for _ in range(len(imagesA))]
    f = open(fileName, "w")
    for i, imageI in enumerate(imagesA):
        print(i)
        for j, imageJ in enumerate(imagesB):
            for (pi, pj) in zip(imageI, imageJ):
                if pi and pj:
                    dot_table[i][j] += pi * pj
            f.write(str(dot_table[i][j])+",")
        f.write("\n")

# gen_table(train_images, train_images, "./dot_table_train")
# gen_table(test_images, train_images, "./dot_table_test_train")
# gen_table(test_images, test_images, "./dot_table_test")

""" Load dot table """
dot_table_train = [[float(i) for i in line.strip().split(",")] for line in open("./dot_table_train")]
dot_table_test_train = [[float(i) for i in line.strip().split(",")] for line in open("./dot_table_test_train")]
dot_table_test = [[float(i) for i in line.strip().split(",")] for line in open("./dot_table_test")]


""" Define self-designed kernel """
def K(a, b, gamma=0.0078125, train=True):
    if train:
        return dot_table_train[a][b] + math.exp(-gamma * (dot_table_train[a][a] - 2*dot_table_train[a][b] + dot_table_train[b][b]))
    return dot_table_test_train[a][b] +  math.exp(-gamma * (dot_table_test[a][a] - 2*dot_table_test_train[a][b] + dot_table_train[b][b]))

""" Generate precompute-kernel """
f = open("./data/precompute-kernel-train", "w")
# f = open("./data/precompute-kernel-test", "w")
for i, label in enumerate(train_labels):
# for i, label in enumerate(test_labels):
    print(i)
    f.write("%d %d:%d" % (label, 0, i+1))
    for j in range(len(train_images)):
        f.write(" %d:%.3f" % (j+1, K(i, j)))
        # f.write(" %d:%.3f" % (j+1, K(i, j, train=False)))
    f.write("\n")
