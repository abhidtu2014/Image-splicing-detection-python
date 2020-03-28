# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:53:35 2018

@author: Abhishek
"""
import cv2
import matplotlib.pyplot as plt
def gradient(img):
  #img=cv2.imread('spliced.jpg')

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  k = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))   
    
  #erosion = cv2.erode(img, k, iterations = 1)
    
  #dilation = cv2.dilate(img, k, iterations = 1)
        
  gradientImg = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)
  
  #dilatedg= cv2.dilate(gradientImg, k, iterations = 1)
  #output = [img, erosion, dilation, gradientImg]

  return gradientImg
    
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt 
   
output = [img, erosion, dilation, gradient]
    
titles = ['Original', 'Erosion', 'Dilation', 'Gradient']
    
for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(output[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

plt.show() 
"""