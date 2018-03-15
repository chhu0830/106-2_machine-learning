#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from random import random
import sys

n = int(sys.argv[1])
p = [float(i) for i in sys.argv[2:]]
p.reverse()
for i in range(n):
    x = random() * n * 100
    y = sum([pi * x**i for i, pi in enumerate(p)]) + (random() * 10 - 5)
    print "%f,%f" % (x, y)
