#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def images(file_name):
    with open(file_name, "rb") as raw_data:
        raw_data.read(4)
        n = int.from_bytes(raw_data.read(4), byteorder="big")
        r = int.from_bytes(raw_data.read(4), byteorder="big")
        c = int.from_bytes(raw_data.read(4), byteorder="big")
        images = [raw_data.read(r * c) for i in range(n)]
    return n, r, c, images

def labels(file_name):
    with open(file_name, "rb") as raw_data:
        raw_data.read(4)
        n = int.from_bytes(raw_data.read(4), byteorder="big")

        labels = [raw_data.read(1)[0] for i in range(n)]
    return n, labels
