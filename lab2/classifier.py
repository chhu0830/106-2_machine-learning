#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys

def normal(x, mean, var):
    if var == 0:
        var = 0.2
    return math.e**-((x - mean)**2 / (2.0 * var)) / (2.0 * math.pi * var)**0.5

class Naive():
    def __init__(self, label_n, pixel_n, bin_n=32, option=0):
        self.label_count = [0 for i in range(label_n)]
        self.label_n = label_n
        self.pixel_n = pixel_n
        self.bin_n   = bin_n
        self.option  = option

    def fit(self, images, labels):
        self.total = len(labels)
        if self.option == 0:
            self.table = [[[0 for k in range(self.bin_n)] for j in range(self.pixel_n)] for i in range(self.label_n)]
            for i, image in enumerate(images):
                self.label_count[labels[i]] += 1
                for j, pixel in enumerate(image):
                    self.table[labels[i]][j][int(pixel / (256 / self.bin_n))] += 1
        else:
            self.table = [[{"mean": 0, "var": 0} for j in range(self.pixel_n)] for i in range(self.label_n)]
            for i, image in enumerate(images):
                self.label_count[labels[i]] += 1
                for j, pixel in enumerate(image):
                    self.table[labels[i]][j]["mean"] += pixel
                    self.table[labels[i]][j]["var"]  += pixel**2
            for i in range(self.label_n):
                for j in range(self.pixel_n):
                    self.table[i][j]["mean"] /= self.label_count[i]
                    self.table[i][j]["var"]   = float(self.table[i][j]["var"]) / float(self.label_count[i]) - self.table[i][j]["mean"]**2
    
    def score(self, images, labels):
        if self.option == 0:
            correct = 0
            for k, image in enumerate(images):
                probabilities = []
                for i in range(self.label_n):
                    probabilities.append(1)
                    for j, pixel in enumerate(image):
                        probabilities[-1] *= self.table[i][j][int(pixel / (256 / self.bin_n))] or 1
                    probabilities[-1] *= self.label_count[i]
                # print(probabilities)
                m = 0
                for i in range(self.label_n):
                    if m == i:
                        continue
                    if probabilities[m] * self.label_count[i]**self.pixel_n < probabilities[i] * self.label_count[m]**self.pixel_n:
                        m = i
                if m == labels[k]:
                    correct += 1
            return float(correct) / float(len(labels))
        else:
            correct = 0
            for k, image in enumerate(images):
                probabilities = []
                for i in range(self.label_n):
                    probabilities.append(0.0)
                    for j, pixel in enumerate(image):
                        p = normal((pixel - self.table[i][j]["mean"]) / (self.table[i][j]["var"]**0.5 or 0.1), 0, 1)
                        probabilities[-1] += math.log(p or 0.01)
                    probabilities[-1] += math.log(float(self.label_count[i]) / float(self.total))
                # print(labels[k], probabilities)
                if max((v, i) for i, v in enumerate(probabilities))[1] == labels[k]:
                    correct += 1
            return float(correct) / float(len(labels))
