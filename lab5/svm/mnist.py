#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from svmutil import *
import sys

""" libsvm-format is generate by libsvm-format.py """
train_labels, train_images = svm_read_problem("../../../data/libsvm-format-train")
test_labels, test_images = svm_read_problem("../../../data/libsvm-format-test")

prob = svm_problem(train_labels, train_images)
param = svm_parameter("-q")
param_best = svm_parameter("-c 32 -g 0.0078125 -q")
param_linear = svm_parameter("-t 0 -q")
param_poly = svm_parameter("-t 1 -q")

model = svm_train(prob, param)
svm_predict(test_labels, test_images, model)

""" precompute-kernel in generate by precompute-kernel.py """
pre_train_labels, pre_train_images = svm_read_problem("../../../data/precompute-kernel-train")
pre_test_labels, pre_test_images = svm_read_problem("../../../data/precompute-kernel-test")

print("File loaded")
prob_pre = svm_problem(pre_train_labels, pre_train_images, isKernel=True)
param_pre = svm_parameter("-t 4")

model = svm_train(prob_pre, param_pre)
svm_predict(pre_test_labels, pre_test_images, model)

"""
linear 95.08%
polynomial 34.68%
radial basis 95.32%
best 98.04%
linear+RBF 95.2%
"""

