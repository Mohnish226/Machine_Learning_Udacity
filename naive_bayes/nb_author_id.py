#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
    
    Results:
    no. of Chris training emails: 7936
    no. of Sara training emails: 7884
    Time To Train: 2.741
    Time To Predict: 0.0
    Accuracy : 0.973265073948
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
t0 = time()

#Train
clf.fit(features_train,labels_train)
print("Time To Train:", round(time()-t0, 3))

#Predict
pred = clf.predict(features_test)
t1 = time()
print("Time To Predict:",round(time()-t1, 3))

#Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(labels_test, pred))

#########################################################


