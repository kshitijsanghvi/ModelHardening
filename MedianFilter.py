# -*- coding: utf-8 -*-
"""
@author: kshit
"""

import cv2
import os
import matplotlib.pyplot as plt
path = "C:\\Users\\kshit\\Desktop\\" 
os.chdir(path)
img = cv2.imread("sp.jfif")
plt.imshow(img)
img_median = cv2.medianBlur(img, 5)
plt.imshow(img_median)
