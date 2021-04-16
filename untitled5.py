# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:46:29 2021

@author: kshit
"""

# -*- coding: utf-8 -*-
"""
@author: kshit
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

os.chdir("C:\\Users\\kshit\\Desktop")
for file in glob.glob("*.png"):
    img = cv2.imread(file)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (10,10,20,20)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    filename = "foreground"+file
    cv2.imwrite(filename,img)
