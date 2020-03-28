# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 00:07:58 2018

@author: Abhishek
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Gradient_image import gradient
import scipy as sp
import skimage.color
import skimage.io

def com(image):
   
   gradientImg=gradient(image)
   dilatedGradient=cv2.dilate(gradientImg, np.ones((5,5), np.uint8) , iterations=1)
   gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

   w,h= gray.shape[:2]
   
   yc = np.sum(dilatedGradient.mean(axis=0))
   xc  =np.sum(dilatedGradient.mean(axis=1))
   
   x=xc/w
   y=yc/h
   
   if y>w:
       y=w
       
   if x>h:
      x=h
   #plt.imshow(image)
   #plt.scatter([x],[y],c='r')
   
   ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 
   (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    
   cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
  
   (_,radius)=cv2.minEnclosingCircle(cnts[0])
  
  
   return ((x,y),radius)


#Earlier version

"""
  image=cv2.imread('spliced7.jpg')
  
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

  ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 
  (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    
  cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
  
  (_,radius)=cv2.minEnclosingCircle(cnts[0])
  
  return ((x,y),radius) 
  
"""
   