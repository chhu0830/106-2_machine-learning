#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from svmutil import *
import sys
import numpy as np
sys.path.append('../../')
from utils import pca, preprocess

''' libsvm-format is generate by libsvm-format.py '''
train_labels, train_images = svm_read_problem('../../../lab5/data/libsvm-format-train')
test_labels, test_images = svm_read_problem('../../../lab5/data/libsvm-format-test')

''' setting parameters according to the README '''
prob = svm_problem(train_labels, train_images)
param = svm_parameter('-q')
param_best = svm_parameter('-c 32 -g 0.0078125 -q')
param_linear = svm_parameter('-t 0 -q')
param_poly = svm_parameter('-t 1 -g 1 -q')
param_rbf  = svm_parameter('-g 0.0078125 -q')

model = svm_train(prob, param)


"""
''' precompute-kernel in generate by precompute-kernel.py '''
pre_train_labels, pre_train_images = svm_read_problem('../../../lab5/data/precompute-kernel-train')
pre_test_labels, pre_test_images = svm_read_problem('../../../lab5/data/precompute-kernel-test')

print('File loaded')
prob_pre = svm_problem(pre_train_labels, pre_train_images, isKernel=True)
param_pre = svm_parameter('-t 4')

model = svm_train(prob_pre, param_pre)
"""

''' get support vectors '''
n = model.get_sv_indices()

''' draw support vectors and dots in 2D space with PCA '''
images, labels = preprocess(path='../../../lab5/data/')
pca(images, labels, '../../results/svm-linearbf.png', special=[image for i, image in enumerate(images) if i in n])
