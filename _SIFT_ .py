# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:16:03 2018

@author: Abhishek
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Gradient_image import gradient
from sklearn.decomposition import PCA
import itertools
from centreOfMass import com
  
def sift(image):
  #image=cv2.imread('dvmm3.tif')
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  
  sift= cv2.xfeatures2d.SIFT_create()
  
  #compute SIFT descriptors
  kps, desc = sift.detectAndCompute(gray, None)
      
  if len(kps)==0:
      return np.zeros(1200)
  
  #Applying PCA
  pca = PCA(n_components = 8)
  desc = pca.fit_transform(desc)
    
 # desc=desc.astype("float64")
  new_desc=[]
  for de in desc:
      x=[]
      for val in de:
          x.append(val)
      l=len(x)    
      while l<8:
          x.append(0.0)
          l+=1
      new_desc.append(x)
      
          
  (center,_)=com(image)
  
  def distance(x,y):   
    return np.sqrt(np.sum((x-y)**2))
  
  #w,h= gray.shape[:2]
   
  d=[]
  #check=np.zeros(w*h)
  
  index=0
  for p in kps:
      p=p.pt
      dist = distance(np.array(center),np.array(p))
   #   num=int(dist)
    #  if check[num]==0:
     #   check[num]=1
      d.append((dist,index))
      index=index+1

  d.sort()
    
  #final_kps=[]
  features = [] 
  ctr=0
#Creating Final feature descriptor
#To choose How much Keypoints you want
  while ctr<150:
    for x in d:
        if ctr<150:
            #final_kps.append(kps[x[1]])
            features.append(new_desc[x[1]])
            ctr=ctr+1
        else:
            break
 
  #Visualizing keypoints on image
  #plt.imshow(cv2.drawKeypoints(image, final_kps, image))
  #plt.scatter([center[0]],[center[1]],c='r')
  #plt.show()    
  
  flattenFeatures= list(itertools.chain(*features))
           
  return flattenFeatures
  #return desc