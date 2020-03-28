# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:02:13 2018

@author: Abhishek
"""

import numpy as np
from glob import glob
import mahotas
import mahotas.features
import milk
from zernike import Zernike
from _SIFT_ import sift
from timeit import default_timer as timer
from sklearn.decomposition import KernelPCA
import scipy as sp
import cv2
import matplotlib.pyplot as plt
from lbp import LBP
from TAS import tas
from sklearn.decomposition import PCA


positives = glob('positives/*.jpg') # Authentic images
negatives = glob('negatives/*.jpg') #spliced images


Features=[] #Final Feature matrix
for im in (positives+negatives):
     img = cv2.imread(im) # read image one by one
          
     features=[]  # features for current image
     features=list(mahotas.features.haralick(img).mean(0)) #haralick
     features+=list(Zernike(img)) #zernike
     features+=(list(sift(img)))  #sift
     Features.append(features)
   
 
#Apply KPCA for feature reduction
# will select best 440 features out of one image
kpca = KernelPCA(n_components = 440, kernel='rbf') 
Features = kpca.fit_transform(Features)


labels =  [1] * len(positives) + [0] * len(negatives) #Labels 

#creating SVM Classifier
learner = milk.defaultclassifier()
#model = learner.train(Features, labels)


start=timer()
cm,names,preds = milk.nfoldcrossvalidation(Features,labels,nfolds=10,classifier=learner, return_predictions=True)
end=timer()
print(end-start) #Time taken to perform validation

TP=cm[0][0] # class Spliced, predicted as Spliced
TN=cm[1][1] #class Authentic, predicted as Authentic
FN=cm[0][1] #class Spliced, predicted as Authentic
FP=cm[1][0] #class Authentic, predicted as Spliced

print("Accuracy=", ((TN+TP)/float(TN+TP+FP+FN))*100.00)
print("Precision=", (TP/float(TP+FP))*100.00)
print("Recall=" , (TP/float(TP+FN))*100.00)

"""
#Xgboost with KFold-cross Validation
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
model = XGBClassifier()
kfold = KFold(n_splits=10, random_state=0)
results = cross_val_score(model, Features, labels, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


#Xgboost with Stratified KFold-cross Validation
from sklearn.model_selection import StratifiedKFold
model = XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(model, Features, labels, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
"""