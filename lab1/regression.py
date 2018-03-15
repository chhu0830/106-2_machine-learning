#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matrix import Matrix

class LSE():
    def __init__(self, bases, lmd=0):
        self.bases = bases
        self.lmd = lmd

    def fit(self, x, y):
        A = []
        for i in range(self.bases):
            A.append([float(xi)**i for xi in x])
        A.reverse()
        self.A = Matrix(A).transpose()
        self.b = Matrix([y]).transpose()
        self.w = (self.A.transpose() * self.A + Matrix(self.bases) * self.lmd).inverse() * (self.A.transpose() * self.b)
        return self.w.transpose().data[0]

    def show(self):
        w = self.w.transpose().data[0]
        for i in range(self.bases-1):
            print("(%.3f x^%d) + " % (w[i], self.bases-i-1), end='')
        print("(%.3f) = y" % w[-1])

    def score(self):
        return round(((self.A * self.w - self.b).transpose() * (self.A * self.w - self.b)).data[0][0], 3)

class Newton():
    def __init__(self, bases, lmd=0, err=1):
        self.bases = bases
        self.lmd = lmd
        self.err = err

    def fit(self, x, y):
        A = []
        for i in range(self.bases):
            A.append([float(xi)**i for xi in x])
        A.reverse()
        self.A = Matrix(A).transpose()
        self.b = Matrix([y]).transpose()

        w0 = Matrix([[0 for i in range(self.bases)]]).transpose()
        while True:
            w1 = w0 - self.hession(w0).inverse() * self.gradient(w0)
            if abs(self.error(w1).data[0][0] - self.error(w0).data[0][0]) < self.err:
                break
            w0 = w1
        self.w = w1
        return self.w.transpose().data[0]

    def show(self):
        w = self.w.transpose().data[0]
        for i in range(self.bases-1):
            print("(%.3f x^%d) + " % (w[i], self.bases-i-1), end='')
        print("(%.3f) = y" % w[-1])

    def score(self):
        return round(self.error(self.w).data[0][0], 3)

    def error(self, x):
        return (self.A * x - self.b).transpose() * (self.A * x - self.b)

    def gradient(self, x):
        return self.A.transpose() * self.A * 2 * x - self.A.transpose() * self.b * 2

    def hession(self, x):
        return self.A.transpose() * self.A * 2

