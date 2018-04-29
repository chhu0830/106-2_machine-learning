#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
sys.path.append("../")
from lab1.matrix import Matrix
from lab3.generator import Gaussian

def logestic(x):
    return 1.0 / (1+math.e**-x)

n  = int(sys.argv[1])
x1 = [Gaussian(int(sys.argv[2]), int(sys.argv[3])).gen() for i in range(n)]
y1 = [Gaussian(int(sys.argv[4]), int(sys.argv[5])).gen() for i in range(n)]
x2 = [Gaussian(int(sys.argv[6]), int(sys.argv[7])).gen() for i in range(n)]
y2 = [Gaussian(int(sys.argv[8]), int(sys.argv[9])).gen() for i in range(n)]

phi= Matrix([x1+x2, y1+y2, [1 for _ in range(n*2)]]).transpose()
y  = Matrix([[0 for _ in range(n)]+[1 for _ in range(n)]]).transpose()
w0 = Matrix([[0, 0, 0]]).transpose()

alpha = 0.001
threshold = 0.01
while True:
    gradient = phi.transpose() * (Matrix([[logestic((Matrix([phi.data[i]])*w0).data[0][0]) for i in range(n*2)]]).transpose() - y)
    D = Matrix(n*2)
    for i in range(n*2):
        z = logestic((Matrix([phi.data[i]])*w0).data[0][0])
        D.data[i][i] = z * (1-z)
    hessian = phi.transpose() * D * phi
    try:
        w1 = w0 - hessian.inverse() * gradient
    except ZeroDivisionError:
        w1 = w0 - gradient * alpha
    s = j = 0
    for i in range(n*2):
        z = logestic((Matrix([phi.data[i]])*w1).data[0][0])
        j += y.data[i][0]*math.log(z+0.00001) + (1-y.data[i][0])*math.log(1-z+0.00001)
        s += int(z+0.5) == int(y.data[i][0])
    print("J = %f, Acc = %f, w = " % (j, (s/float(n*2))), w1.transpose().data[0], sep="")
    # all gradient small than threshold
    if sum([abs(v) > threshold for v in (w1-w0).transpose().data[0]]) == 0:
        break
    w0 = w1
