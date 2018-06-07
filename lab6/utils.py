#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
import os.path

def pca(data, label, filename=None, color=['r', 'g', 'b', 'c', 'y'], special=None):
    ''' get the covariance of data and get the eigenvectors of the covariance matrix '''
    cov = np.cov(np.array(data).T)
    w, v = LA.eig(cov)

    ''' sort the eigenvectors with corresponding eigenvalues from large to small '''
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:,idx]

    ''' project data onto first two eigenvectors '''
    for i, X in enumerate(data):
        x = np.dot(v[:,0], X)
        y = np.dot(v[:,1], X)
        plt.scatter(x, y, c=color[label[i]], s=5)

    ''' mark the special data points with black triangle sign (for support vectors) '''
    if special:
        for s in special:
            x = np.dot(v[:,0], s)
            y = np.dot(v[:,1], s)
            plt.scatter(x, y, c="black", s=5, marker='v')

    ''' save the figure and show it '''
    if filename:
        plt.savefig(filename)
    plt.show()

def lda(data, labels, filename=None, color=['r', 'g', 'b', 'c', 'y']):
    ''' calculate according to the fomula '''
    K = len(np.unique(labels))
    M = np.average(data, axis=0)
    m = [np.average([d for (d, l) in zip(data, labels) if l==k], axis=0) for k in range(K)]
    N = len(data[0])

    Sw = np.zeros((N, N))
    for (d, l) in zip(data, labels):
        Sw += np.dot(np.array(d-m[l], ndmin=2).T, np.array(d-m[l], ndmin=2))
    Sb = np.zeros((N, N))
    for j in range(K):
        Sb += labels.count(j) * np.dot(np.array(m[j]-M, ndmin=2).T, np.array(m[j]-M, ndmin=2))
    w, v = LA.eig(np.dot(LA.pinv(Sw), Sb))

    ''' sort the eigenvectors with corresponding eigenvalues from large to small '''
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:,idx]

    ''' project data onto first two eigenvectors '''
    for i, X in enumerate(data):
        x = np.dot(v[:,0], X)
        y = np.dot(v[:,1], X)
        plt.scatter(x, y, s=5, c=color[label[i]])

    ''' save the figure and show it '''
    if filename:
        plt.savefig(filename)
    plt.show()

''' read data from file '''
def preprocess(path='../lab5/data/', t='train'):
    X_train = [[float(p) for p in line.strip().split(',')] for line in open(path+'X_'+t+'.csv')]
    T_train = [int(line)-1 for line in open(path+'T_'+t+'.csv')]
    return X_train, T_train

''' precompute dot calculation '''
def dot_table(set1, set2, filename):
    if os.path.isfile(filename):
        return np.load(open(filename, "rb"))
    table = [[np.dot(i, j) for j in set2] for i in set1]
    np.save(open(filename, "wb"), table)
    return table

if __name__ == '__main__':
    data, label = preprocess()
    pca(data, label, './results/pca.png')
    lda(data, label, './results/lda.png')
