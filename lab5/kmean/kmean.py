#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

""" Read data from file """
circles = [[float(x) for x in line.strip().split(",")] for line in open("../data/circle.txt")]
moons = [[float(x) for x in line.strip().split(",")] for line in open("../data/moon.txt")]

""" Dot table is generate by dot_table.py, dot_table[a][b] represent inner product of a and b """
dot_table_circles = [[float(x) for x in line.split(",")] for line in open("./dot_table_circle")]
dot_table_moons = [[float(x) for x in line.split(",")] for line in open("./dot_table_moon")]


""" Define kernel """
def linear(a, b):
    return dot_table[a][b]

def rbf(a, b):
    gamma = (len(sys.argv) > 4 and float(sys.argv[4])) or 20
    return math.exp(-gamma * (dot_table[a][a] - 2*dot_table[a][b] + dot_table[b][b]))

class Kmean():
    def __init__(self, kernel="linear", K=2, C=["r", "y", "c", "b"]):
        self.kernel = Kernel[kernel]
        self.K = K
        self.C = C
        self.path = "../results/%s-%s-%s" % (sys.argv[1], sys.argv[2], (sys.argv[3] + sys.argv[4] if len(sys.argv) > 4 else ""))
        try:
            os.stat(self.path)
        except:
            os.mkdir(self.path)

    def fit(self, X):
        self.X = X
        """ Random initialization """
        self.labels = [random.choice(range(self.K)) for i in range(len(X))]
        """ Non random initialization """
        # self.path += "-notrand"
        # p = 0.6
        # self.labels = [np.random.choice(np.arange(0, 2), p=(p, 1-p)) if math.sqrt(x[0]*x[0] + x[1]*x[1]) < 0.6 else np.random.choice(np.arange(0, 2), p=(1-p, p)) for x in X]
        # self.labels = [np.random.choice(np.arange(0, 2), p=(p, 1-p)) if x[1] < 0.25 else np.random.choice(np.arange(0, 2), p=(1-p, p)) for x in X]
        self.show(0)

        c = 0
        while True:
            c += 1
            print(c)
            labels = []
            clusters = [[i for i, label in enumerate(self.labels) if label==k] for k in range(self.K)]
            mutual  = [sum([sum([self.kernel(y, z) for z in clusters[k]]) for y in clusters[k]]) / len(clusters[k])**2 for k in range(self.K)]
            """ For each point, caculate in which cluster should the point be """
            for x in range(len(X)):
                dis = [ self.kernel(x, x)
                        - 2.0 * sum([self.kernel(x, y) for y in clusters[k]]) / len(clusters[k])
                        + mutual[k]
                        for k in range(self.K) ]
                labels.append(dis.index(min(dis)))
            """ If all points are in the same cluster of last round, then converged """
            if sum([a != b for (a, b) in zip(self.labels, labels)]) == 0:
                self.show(c)
                break
            self.labels = labels
            # self.show(c)

    """ Draw points to figure """
    def show(self, i):
        for (x, label) in zip(self.X, self.labels):
            plt.scatter(x[0], x[1], c=self.C[label])

        plt.savefig("%s/%d.png" % (self.path, i))
        # plt.show()

Kernel = {"linear": linear, "rbf": rbf}
Points = {"circle": circles, "moon": moons}
Dot_table = {"circle": dot_table_circles, "moon": dot_table_moons}

points = Points[sys.argv[1]]
dot_table = Dot_table[sys.argv[1]]

kmean = Kmean(K=int(sys.argv[2]), kernel=sys.argv[3])
kmean.fit(points)
