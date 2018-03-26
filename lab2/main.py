#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import preprocess
import sys
from classifier import Naive

training_images_file = (len(sys.argv) >= 2 and sys.argv[1]) or "data/train-images-idx3-ubyte"
training_labels_file = (len(sys.argv) >= 3 and sys.argv[2]) or "data/train-labels-idx1-ubyte"
testing_images_file  = (len(sys.argv) >= 4 and sys.argv[3]) or "data/t10k-images-idx3-ubyte"
testing_labels_file  = (len(sys.argv) >= 5 and sys.argv[4]) or "data/t10k-labels-idx1-ubyte"

n, r, c, training_images = preprocess.images(training_images_file)
n, training_labels = preprocess.labels(training_labels_file)

naive_bayes = Naive(10, r * c, option=1)
naive_bayes.fit(training_images, training_labels)

n, r, c, testing_images = preprocess.images(testing_images_file)
n, testing_labels = preprocess.labels(testing_labels_file)

print(naive_bayes.score(testing_images, testing_labels))
