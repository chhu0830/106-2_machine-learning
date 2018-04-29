#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys
import random
sys.path.append("../")
import lab2.preprocess as preprocess

print("Fetching data")
N, R, C, images = preprocess.images("../lab2/data/t10k-images-idx3-ubyte")
_, labels = preprocess.labels("../lab2/data/t10k-labels-idx1-ubyte")
D = R*C
K = 10

N = int(N/1)
images = images[:N]
labels = labels[:N]
print([labels.count(i) for i in range(10)])

print("Initiating")
tmp = [[0 for _ in range(D)] for _ in range(N)]
count = [labels.count(i) for i in range(K)]
mu = [[random.uniform(0.4, 0.6) for _ in range(D)] for _ in range(K)]
for i, image in enumerate(images):
    for j, pixel in enumerate(image):
        # mu[labels[i]][j] += (pixel>>7) / float(count[labels[i]])
        tmp[i][j] = pixel>>7
pi = [0.1 for _ in range(K)]

images = tmp
threshold = 0.01
c = 0
while True:
    print("round", c)
    print("E step")
    t = time.time()
    z = [[pi[k] for k in range(K)] for _ in range(N)]
    for n in range(N):
        for k in range(K):
            for i in range(D):
                # z[n][k] *= mu[k][i]**(images[n][i]) * (1-mu[k][i])**(1-images[n][i])
                z[n][k] *= mu[k][i] if images[n][i] else (1-mu[k][i])
        s = sum(z[n])
        for k in range(K):
            z[n][k] /= s
    print(time.time() - t)

    print("M step")
    t = time.time()
    Nm = [sum([z[n][m] for n in range(N)]) for m in range(K)]
    flag = True
    for m in range(K):
        for i in range(D):
            # mu[m][i] = sum([z[n][m]*(images[n][i]) for n in range(N)]) / Nm[m]
            x = sum([z[n][m] if images[n][i] else 0 for n in range(N)]) / Nm[m]
            if abs(mu[m][i] - x) > threshold:
                flag = False
            mu[m][i] = x
        pi[m] = Nm[m] / N
    print(time.time() - t)
    c += 1
    if flag or c > 50:
        break

print("Evaluating")
print("====================")
e = [[0 for _ in range(K)] for _ in range(K)]
for n in range(N):
    e[z[n].index(max(z[n]))][labels[n]] += 1
for j in range(K):
    for i in range(j+1, K):
        if e[i][j] > e[j][j]:
            e[j], e[i] = e[i], e[j]
for i in e:
    for j in i:
        print("%5d" % j, end=" ")
    print()
for i in range(K):
    TP = e[i][i]
    FP = sum(e[i]) - e[i][i]
    FN = count[i] - e[i][i]
    TN = sum(count) - count[i] - sum(e[i]) + e[i][i]
    print("Confusion table of", i)
    print("TP:%4d, FP:%4d, FN:%4d, TN:%4d, Sensitivity:%5.2f, Specificity:%5.2f" % (TP, FP, FN, TN, round(TP / float(TP+FN) * 100.0, 1), round(TN / float(TN+FP) * 100.0, 1)))
    print("====================")
