#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:56:34 2018

@author: jarrarhaider
"""

import os
import codecs
from shutil import copyfile
from math import floor
import in_place
from nltk.corpus import stopwords

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.xml'), all_files))
    return data_files

from random import shuffle

def randomize_files(file_list):
    shuffle(file_list)
    


def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

data_files = get_file_list_from_dir("citations_class")

randomize_files(data_files)

training,testing  = get_training_and_testing_sets(data_files)

for training_files in training:
    copyfile('citations_class/' + training_files , "training/" + training_files)
    with in_place.InPlaceText('training/' + training_files , encoding = 'cp1252') as file:
        for line in file:
            line = line.replace('"id=', 'id="')
            line = line.replace('&','')
            file.write(line)
for testing_files in testing:
    copyfile('citations_class/' + testing_files , "testing/" + testing_files)
    with in_place.InPlaceText('testing/' + testing_files , encoding = 'cp1252') as file:
        for line in file:
            line = line.replace('"id=', 'id="')
            line = line.replace('&','')
            file.write(line)
    
    
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.xml'), all_files))
    return data_files

data_files = get_file_list_from_dir("training")

for file in data_files:
    with codecs.open('training/'+file,'r',encoding='cp1252') as inFile, codecs.open('training_preprocessed/'+file,'w',encoding='cp1252') as outFile:
        for line in inFile.readlines():
            print(" ".join([word for word in line.lower().translate(str.maketrans('', '', '')).split() 
            if len(word) >=4 and word not in stopwords.words('english')]), file=outFile)


data_files = get_file_list_from_dir("testing")
    
for file in data_files:
    with codecs.open('testing/'+file,'r',encoding='cp1252') as inFile, codecs.open('testing_preprocessed/'+file,'w',encoding='cp1252') as outFile:
        for line in inFile.readlines():
            print(" ".join([word for word in line.lower().translate(str.maketrans('', '', '')).split() 
            if len(word) >=4 and word not in stopwords.words('english')]), file=outFile)