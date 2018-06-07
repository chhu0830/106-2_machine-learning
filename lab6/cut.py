#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import pca, preprocess, dot_table
import numpy as np
from numpy import linalg as LA
from time import time
import os.path
import sys

''' parameters '''
TYPE = ['ratio', 'normalized'][0]               # ratio | normalized
KERNEL = ['linear', 'rbf', 'linearbf'][0]       # linear | rbf | linearbf
GAMMA = 0.001                                   # gamma value for rbf
LOAD = [True, False][1]                         # load the eigenvector from file or noe
K = 5                                           # amount of clusters
FILENAME = None                                 # save the result to file


''' define kernels '''
def linear(i, j, table):
    return table[i][j]

def rbf(i, j, table, gamma=GAMMA):
    return np.exp(-gamma * (table[i][i]-2*table[i][j]+table[j][j]))

def linearbf(i, j, table, gamma=GAMMA):
    return linear(i, j, table) + rbf(i, j, table, gamma)

def kmean(data, K):
    ''' random choose K data points as centers '''
    center = data[np.random.choice(len(data), 5)]
    while True:
        labels = []
        cluster = [np.zeros(len(data[0])) for _ in range(K)]
        count  = [0 for _ in range(K)]
        for d in data:
            ''' find the closest center '''
            dis = [np.dot((d-c), (d-c)) for c in center]
            label = np.argmin(dis)
            labels.append(label)
            cluster[label] += d.real
            count[label] += 1
        print('Each cluster:', count)
        
        ''' Update centers and check whether it has converged '''
        cluster = [cluster[i] / float(count[i]) for i in range(K)]
        if np.array_equal(center, cluster):
            break
        center = cluster

    return labels

print('Fetching data ...')
t = time()
X_train, T_train = preprocess()
table = dot_table(X_train, X_train, 'dot_table.npy')
print('Time:', time() - t)

if LOAD and os.path.isfile('w.npy') and os.path.isfile('v.npy'):
    print('Loading eigenvector ...')
    t = time()
    w = np.load(open('w.npy', 'rb'))
    v = np.load(open('v.npy', 'rb'))
    print('Time:', time() - t)

else:
    kernel = {'linear':linear, 'rbf':rbf, 'linearbf':linearbf}[KERNEL]
    print('Generating L ...')
    t = time()
    N = len(table)
    W = [[kernel(i, j, table) for j in range(N)] for i in range(N)]
    D = [[sum(W[i]) if i==j else 0 for j in range(N)] for i in range(N)]
    L = np.array(D) - np.array(W)
    print('Time:', time() - t)

    print('Calculating eigenvector ...')
    t = time()
    ''' use second method provided on the hangout for normalized cut '''
    w, v = LA.eig(L) if TYPE=='ratio' else LA.eig(np.dot(LA.inv(D), L))
    np.save(open('w.npy', 'wb'), w)
    np.save(open('v.npy', 'wb'), v)
    print('Time:', time() - t)

idx = np.argsort(w)
w = w[idx]
v = v[:,idx]
print('Eigenvalue:', w[:4])

U = np.array([v[:,u] for u in range(1, K+1)]).T

print('Kmean ...')
labels = kmean(U, K)
pca(X_train, labels, FILENAME)
