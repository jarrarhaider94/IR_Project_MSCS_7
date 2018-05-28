#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 23:43:46 2018

@author: jarrarhaider
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import os
import sys
from IPython.display import display , Image
import PIL 
import pandas as pd
from scipy import ndimage
from six.moves import cPickle as pickle
import tensorflow as tf
import scipy.io
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from datetime import datetime as dt
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report





db_world_matlab = scipy.io.loadmat('MATLAB/dbworld_bodies_stemmed.mat')
#To view the uploaded file
#print(db_world_matlab)


#Features/Labels in dbworld_bodies_stemmed file
labels = db_world_matlab['labels']
labels=np.asarray(labels)
#To view the labels
#print("Labels : ")
#print (labels)

#Input variable in dbworld_bodies_stemmed file
data = db_world_matlab['inputs']  
data=np.asarray(data)
#To view the labels
#print("Data : ")
#print (data)

#As we don't know the dimension, we just give provide the shape of label for first index.
#This way we don't have to look and insert the value manually.

X = pd.DataFrame(data)
y = labels.ravel()



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state = 39)

def Rocchio_Algorith():
    clf = NearestCentroid()
    clf.fit(X_train, Y_train)
    
    pred_result = clf.predict(X_test)

    print (pred_result)
    print()
    print (Y_test)

    print ('classification report: ')
#     print classification_report(y_test, yhat)
    print(classification_report(Y_test, pred_result))
    
    print ('f1 score')
    print (f1_score(Y_test, pred_result, average='macro'))
    
    print ('accuracy score')
    print (accuracy_score(Y_test, pred_result))
    
    precision = precision_score(Y_test, pred_result, average=None)
    print ("Precision : ")
    print (precision)
    recall = recall_score(Y_test, pred_result, average=None)
    print ("Recall : ")
    print (recall)

def Naive_Bayes_Algorith():
    
    mn = MultinomialNB()
    mn.fit(X_train, Y_train)

    pred_result = mn.predict(X_test)

    print (pred_result)
    print()
    print (Y_test)

    print ('classification report: ')
#     print classification_report(y_test, yhat)
    print(classification_report(Y_test, pred_result))
    
    print ('f1 score')
    print (f1_score(Y_test, pred_result, average='macro'))
    
    print ('accuracy score')
    print (accuracy_score(Y_test, pred_result))
    
    precision = precision_score(Y_test, pred_result, average=None)
    print ("Precision : ")
    print (precision)
    recall = recall_score(Y_test, pred_result, average=None)
    print ("Recall : ")
    print (recall)

def KNN_Algorithm():
    
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, Y_train)
    

    pred_result = knn.predict(X_test)

    print (pred_result)
    print()
    print (Y_test)

    print ('classification report: ')
#     print classification_report(y_test, yhat)
    print(classification_report(Y_test, pred_result))
    
    print ('f1 score')
    print (f1_score(Y_test, pred_result, average='macro'))
    
    print ('accuracy score')
    print (accuracy_score(Y_test, pred_result))
    
    precision = precision_score(Y_test, pred_result, average=None)
    print ("Precision : ")
    print (precision)
    recall = recall_score(Y_test, pred_result, average=None)
    print ("Recall : ")
    print (recall)

print ("_____Naive Bayes Classifier_____")
Naive_Bayes_Algorith()

print ("_____Rocchio Classifier_____")
Rocchio_Algorith()

print ("_____KNN Classifier_____")
KNN_Algorithm()