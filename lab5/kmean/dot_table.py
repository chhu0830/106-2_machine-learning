#!/usr/bin/env python3
# -*- coding: utf-8 -*-

circles = [[float(x) for x in line.strip().split(",")] for line in open("./data/circle.txt")]
moons = [[float(x) for x in line.strip().split(",")] for line in open("./data/moon.txt")]

f = open("./dot_table_moon", "w")
dot_table = [[0 for _ in range(len(moons))] for _ in range(len(moons))]
for i, x in enumerate(moons):
    for j, y in enumerate(moons):
        dot_table[i][j] = x[0]*y[0] + x[1]*y[1]
        f.write(str(dot_table[i][j])+",")
    f.write("\n")

