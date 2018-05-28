#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 17:54:36 2018

@author: jarrarhaider
"""
import os
from shutil import copyfile
from math import floor

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
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

data_files = get_file_list_from_dir("labeled_articles")

randomize_files(data_files)

training,testing  = get_training_and_testing_sets(data_files)

for training_files in training:
    copyfile('labeled_articles/' + training_files , "training/" + training_files)
    
for testing_files in testing:
    copyfile('labeled_articles/' + testing_files , "testing/" + testing_files)