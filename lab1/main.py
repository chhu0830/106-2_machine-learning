#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from regression import LSE, Newton

if len(sys.argv) != 4:
    print("usage:", sys.argv[0], "<data file> <bases number> <lambda>")
    exit()

fname = sys.argv[1]
bases = int(sys.argv[2])
lmd   = float(sys.argv[3])

with open(fname) as f:
    data = [line.strip().split(",") for line in f]
x = [datum[0] for datum in data]
y = [datum[1] for datum in data]

lse = LSE(bases=bases, lmd=lmd)
lse.fit(x, y)
print("LSE: ", end="")
lse.show()
print("LSE error:", lse.score())

newton = Newton(bases=bases)
newton.fit(x, y)
print("Newton method: ", end="")
newton.show()
print("Newton method error:", newton.score())
