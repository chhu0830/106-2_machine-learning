#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from generator import Gaussian
import sys

N = int(sys.argv[1])
m = float(sys.argv[2])
s = float(sys.argv[3])

gaussian = Gaussian(m, s)
x = gaussian.gen()
mean = x
m2   = 0
var  = 0
n    = 1
print("%6.3f %6.3f %6.3f" % (x, x, 0))

while n < N:
    x = gaussian.gen()
    n += 1
    d1   = x - mean
    mean = mean + (x - mean)/n
    d2   = x - mean
    m2   = m2 + d1 * d2
    var  = m2 / (n-1)
    print("x =%6.3f, mean =%6.3f, var =%6.3f" % (x, mean, var))
