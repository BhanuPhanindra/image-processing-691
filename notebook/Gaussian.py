# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:22:40 2019

@author: goran
"""
import math
#import cv2

def GaussianBlurImage(image, sigma):
#    pi = math.pi
    result = [[0] * len(image[0]) for i in range(len(image)) ]
    constant = 1 / (2 * math.pi * (sigma) ** 2)
    for i, row in image:
        for j, col in row:
            result[i][j] = constant * math.e ** -( (i**2 + j**2) / (2 * sigma ** 2))
    return result


f = open('./hw1_data/Seattle.jpg', 'r+')
jpegdata = f.read()
#print(jpegdata)