#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matrix import Matrix
from generator import Linear 
import sys


N = int(sys.argv[1])
b = float(sys.argv[2])
a = float(sys.argv[3])
w = [float(w) for w in sys.argv[4:]]
n = len(w)

linear = Linear(w, a)
I = Matrix(n)

s0 = I * b
m0 = Matrix([[0 for i in range(n)]]).transpose()
for _ in range(N):
    x, y = linear.gen()
    X = Matrix([[x**i for i in range(n)]])

    s1 = X.transpose()*X*(1.0/a) + s0
    m1 = s1.inverse() * (X.transpose()*a*y + s0*m0)

    ym = (X*m1).data[0][0]
    yv = (1.0/a) + (X*s1.inverse()*X.transpose()).data[0][0]
    print("x = %f, ym = %f, yv = %f, wm =" % (x, ym, yv), m1.transpose().data[0])

    s0, m0 = s1, m1

    if False:
        import matplotlib.pyplot as plt
        l = Linear(m1.transpose().data[0], 0)
        X = []
        Y = []
        U = []
        L = []
        for i in range(10000):
            x, y = l.gen()
            X.append(x)
            Y.append(y)
            d = Matrix([[x**i for i in range(n)]])
            yv = (1.0/a) + (d*s1.inverse()*d.transpose()).data[0][0]
            U.append(y+yv)
            L.append(y-yv)
        plt.scatter(X, U, c="red", s=1)
        plt.scatter(X, L, c="red", s=1)
        plt.scatter(X, Y, c="blue", s=1)
        plt.show()
