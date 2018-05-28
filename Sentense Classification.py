#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 17:09:59 2018

@author: jarrarhaider
"""

import os
import json
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import NLTKClassifier
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
from sklearn.naive_bayes import MultinomialNB


training_label = []
training_sentences = []

def list_files_from_directory(directory):
    """Lists all file paths from given directory"""

    ret_val = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            ret_val.append(str(directory) + "/" + str(file))
    return ret_val


def read_file(path):
    """Reads all lines from file on given path"""

    f = open(path, "r")
    read = f.readlines()
    ret_val = []
    for line in read:
        if line.startswith("#"):
            pass
        else:
            ret_val.append(line)
    return ret_val


def process_line(line):
    """Returns sentence category and sentence in given line"""

    if "\t" in line:
        splits = line.split("\t")
        s_category = splits[0]
        sentence = splits[1].lower()
        for sw in stopwords:
            sentence = sentence.replace(sw, "")
        pattern = re.compile("[^\w']")
        sentence = pattern.sub(' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return s_category, sentence
    else:
        splits = line.split(" ")
        s_category = splits[0]
        sentence = line[len(s_category)+1:].lower()
        for sw in stopwords:
            sentence = sentence.replace(sw, "")
        pattern = re.compile("[^\w']")
        sentence = pattern.sub(' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return s_category, sentence


def create_json_file(input_folder, destination_file):
    """Writes training data from given folder into formatted JSON file"""

    tr_folder = list_files_from_directory(input_folder)
    all_json = []
    for file in tr_folder:
        lines = read_file(file)
        for line in lines:
            c, s = process_line(line)
            if s.endswith('\n'):
                s = s[:-1]
            json_data = {
                'text': s,
                'label': c
            }
            all_json.append(json_data)
            training_label.append(c)
            training_sentences.append(s)

    with open(destination_file, "w") as outfile:
        json.dump(all_json, outfile)


def prepare_test_data(input_folder):
    """Maps each sentence to it's category"""

    test_folder = list_files_from_directory(input_folder)
    t_sentences = []
    t_categories = []
    for file in test_folder:
        lines = read_file(file)
        for line in lines:
            c, s = process_line(line)
            if s.endswith('\n'):
                s = s[:-1]
            t_sentences.append(s)
            t_categories.append(c)
    return t_categories, t_sentences





def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
    return data_files

from random import shuffle

def randomize_files(file_list):
    shuffle(file_list)
    


# Splitting docs into training and testing folders . This code needs to be run for the first time. But once train and test directories are populated, we need to comment this code out
#def get_training_and_testing_sets(file_list):
#    split = 0.7
#    split_index = floor(len(file_list) * split)
#    training = file_list[:split_index]
#    testing = file_list[split_index:]
#    return training, testing
#
#data_files = get_file_list_from_dir("labeled_articles")
#
#randomize_files(data_files)
#
#training,testing  = get_training_and_testing_sets(data_files)
#
#for training_files in training:
#    copyfile('labeled_articles/' + training_files , "training/" + training_files)
#    
#for testing_files in testing:
#    copyfile('labeled_articles/' + testing_files , "testing/" + testing_files)


#Test Classifier
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

# main

# loading stopwords
input_stopwords = read_file("word_lists/stopwords.txt")
stopwords = []
for word in input_stopwords:
    if word.endswith('\n'):
        word = word[:-1]
        stopwords.append(word)


# prepare training and test data
create_json_file("training", "training.json")
categories, sentences = prepare_test_data("testing")


# Bayes Classifier
#print("Training Naive Bayes Classifier...")
#start_nbc = time.time()
#with open('training.json', 'r') as training:
#    nbc = NaiveBayesClassifier(training, format="json")
#stop_nbc = time.time()
#print("Training Naive Bayes Classifier completed...")
#elapsed = stop_nbc - start_nbc
#print("Training time (in seconds): " + str(elapsed))
#print("Testing Naive Bayes Classifier...")
#correct = 0
#start_nbc = time.time()
#for i in range(0, len(sentences)):
#    category = str(nbc.classify(sentences[i])).lower()
#    expected = str(categories[i]).lower()
#    if category == expected:
#        correct += 1
#stop_nbc = time.time()
#elapsed = stop_nbc - start_nbc
#print("Number of tests: " + str(len(sentences)))
#print("Correct tests: " + str(correct))
#accuracy = correct / len(sentences)
#print("Naive Bayes Classifier accuracy: " + str(accuracy))
#print("Testing time (in seconds): " + str(elapsed))




#Create text vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(training_sentences)
train_mat = vectorizer.transform(training_sentences)
test_mat = vectorizer.transform(sentences)

#Applying TF-IDF to the vector matrix 
tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)
print (train_tfmat.shape)
test_tfmat = tfidf.transform(test_mat)
print (test_tfmat.shape)





#Metrics List
metrics_dict = []
#'name', 'metrics'


mbn_params = {'alpha': [a*0.1 for a in range(0,11)]}
mbn_clf = GridSearchCV(MultinomialNB(), mbn_params, cv=10)
mbn_clf.fit(train_tfmat, training_label)
print ('best parameters')
print (mbn_clf.best_params_)
best_mbn = MultinomialNB(alpha=mbn_clf.best_params_['alpha'])
best_mbn_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, best_mbn)
metrics_dict.append({'name':'Best MultinomialNB', 'metrics':best_mbn_me})

# KNN
for nn in [3]:
    print ('knn with ', nn, ' neighbors')
    knn = KNeighborsClassifier(n_neighbors=nn)
    knn_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, knn)
    metrics_dict.append({'name':'3NN', 'metrics':knn_me})
    print (' ')


#Rocchio Algorithm
    
clf = NearestCentroid()
clf_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, clf)
metrics_dict.append({'name':'Rocchio Algorithm', 'metrics':clf_me})
