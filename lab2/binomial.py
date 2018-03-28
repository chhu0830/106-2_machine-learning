#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

filename = (len(sys.argv) >= 2 and sys.argv[1]) or "data/coin-tossing"
a = (len(sys.argv) >= 3 and sys.argv[2]) or 2
b = (len(sys.argv) >= 4 and sys.argv[3]) or 2

with open(filename) as f:
    events = [event.strip() for event in f]

a0, b0 = a, b
for event in events:
    m = sum([int(e) for e in event])
    n = len(event)
    a1, b1 = m+a0, n-m+b0
    print("MLE:", float(m)/n, "prior:", a0, b0, "post:", a1, b1)
    a0, b0 = a1, b1
