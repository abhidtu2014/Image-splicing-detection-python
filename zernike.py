# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:34:02 2018

@author: Abhishek
"""
import mahotas
import cv2
from watershed_using_scipi import Wsegment
from centreOfMass import com
import imutils
import matplotlib.pyplot as plt
import scipy as sp

def Zernike(image):
  (_cm,_radius)=com(image)
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  moments=mahotas.features.zernike_moments(gray,radius=_radius,cm=_cm)

  return moments
        

#Previous changes and ideas
"""
Watershed will make zernike moments invariant to scaling and transformation
zernike moments are inherently invariant to rotation(But better to use Gray Scale)
#watershed of gray image
#wshed_img=Wsegment(gray_img)
#plt.imshow(wshed_img,cmap='nipy_spectral')
#plt.show()

#inverting the image and finding OTSU threshold
_,thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#auto_canny
edged = imutils.auto_canny(gray_img)

#plt.imshow(edged)
#Finding contours in an image
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
"""

