#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:16:17 2018

@author: jarrarhaider
"""

import xml.etree.ElementTree as ET
from sklearn.naive_bayes import MultinomialNB

import os
import json
from textblob.classifiers import NaiveBayesClassifier
import time
import nltk.classify
from sklearn.svm import LinearSVC
import re
from shutil import copyfile
from sklearn.neighbors.nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt
import pprint as pp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime as dt
from sklearn.grid_search import GridSearchCV


class_label_array = []
text_ = []
training_label = []
training_sentences = []


def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.xml'), all_files))
    return data_files

def get_all_classes_and_labels_from_testing(file_path):
    tree = ET.parse('training_preprocessed/' + file_path)
    root = tree.getroot()
    for class_label in root.iter('class'):
        class_label_array.append(class_label.text)
    for text_label in root.iter('text'):
            text_.append(text_label.text)
    


def list_files_from_directory(directory):
    """Lists all file paths from given directory"""

    ret_val = []
    for file in os.listdir(directory):
        if file.endswith(".xml"):
            ret_val.append(str(directory) + "/" + str(file))
    return ret_val


def prepare_test_data(input_folder):
    """Maps each sentence to it's category"""

    test_folder = list_files_from_directory(input_folder)
    t_sentences = []
    t_categories = []
    
    
    for file in test_folder:
        tree = ET.parse(file)
        root = tree.getroot()
        for class_label in root.iter('class'):
            t_categories.append(class_label.text)
        for text_label in root.iter('text'):
            t_sentences.append(text_label.text)
            
    return t_categories, t_sentences


def create_text_label_array():
    """Writes training data from given folder into formatted JSON file"""
#        
    for x in range(len(text_)):
        training_label.append(class_label_array[x])
        training_sentences.append(text_[x])

    
    
    
    
all_files = get_file_list_from_dir("training_preprocessed")
for files in all_files:
    get_all_classes_and_labels_from_testing(files)
    #print(files)
    
create_text_label_array()
categories, sentences = prepare_test_data("testing_preprocessed")


x = 0
for s in training_sentences:
    if s is None:
        del training_label[x]
    x = x+1


training_sentences = list(filter(None, training_sentences))

z = 0
for sen in sentences:
    if sen is None:
        del categories[z]
    z = z + 1
    
sentences = list(filter(None, sentences))

#Create text vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(training_sentences)
train_mat = vectorizer.transform(training_sentences)
test_mat = vectorizer.transform(sentences)

#Applying TF-IDF to the vector matrix 
tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)
test_tfmat = tfidf.transform(test_mat)



def testClassifier(x_train, y_train, x_test, y_test, clf):

    
    """
    this method will first train the classifier on the training data
    and will then test the trained classifier on test data.
    Finally it will report some metrics on the classifier performance.
    
    Parameters
    ----------
    x_train: np.ndarray
             train data matrix
    y_train: list
             train data label
    x_test: np.ndarray
            test data matrix
    y_test: list
            test data label
    clf: sklearn classifier object implementing fit() and predict() methods
    
    Returns
    -------
    metrics: list
             [training time, testing time, recall and precision for every class, macro-averaged F1 score]
    """
    metrics = []
    start = dt.now()
    clf.fit(x_train, y_train)
    end = dt.now()
    print ('training time: ', (end - start))
    
    # add training time to metrics
    metrics.append(end-start)
    
    start = dt.now()
    yhat = clf.predict(x_test)
    end = dt.now()
    print ('testing time: ', (end - start))
    
    # add testing time to metrics
    metrics.append(end-start)
    
    print ('classification report: ')
#     print classification_report(y_test, yhat)
    pp.pprint(classification_report(y_test, yhat))
    
    print ('f1 score')
    print (f1_score(y_test, yhat, average='macro'))
    
    print ('accuracy score')
    print (accuracy_score(y_test, yhat))
    
    precision = precision_score(y_test, yhat, average=None)
    recall = recall_score(y_test, yhat, average=None)
    
    # add precision and recall values to metrics
    for p, r in zip(precision, recall):
        metrics.append(p)
        metrics.append(r)
    
    
    #add macro-averaged F1 score to metrics
    metrics.append(f1_score(y_test, yhat, average='macro'))
    
    print ('confusion matrix:')
    print (confusion_matrix(y_test, yhat))
    
    # plotting the confusion matrix
    plt.imshow(confusion_matrix(y_test, yhat), interpolation='nearest')
    plt.show()
    
    return metrics


#Metrics List
metrics_dict = []

mbn_params = {'alpha': [a*0.1 for a in range(0,11)]}
mbn_clf = GridSearchCV(MultinomialNB(), mbn_params, cv=10)
mbn_clf.fit(train_tfmat, training_label)
print ('best parameters')
print (mbn_clf.best_params_)
best_mbn = MultinomialNB(alpha=mbn_clf.best_params_['alpha'])
best_mbn_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, best_mbn)
metrics_dict.append({'name':'Best MultinomialNB', 'metrics':best_mbn_me})

#
## KNN
for nn in [10]:
    print ('knn with ', nn, ' neighbors')
    knn = KNeighborsClassifier(n_neighbors=nn)
    knn_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, knn)
    metrics_dict.append({'name':'5NN', 'metrics':knn_me})
    print (' ')


#Rocchio Algorithm
    
clf = NearestCentroid()
clf_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, clf)
metrics_dict.append({'name':'Rocchio Algorithm', 'metrics':clf_me})
    
