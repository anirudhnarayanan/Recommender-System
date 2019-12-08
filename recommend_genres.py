#!/usr/bin/env python

from sklearn.decomposition import PCA
from ml import LogisticRegression
from past.builtins import xrange
import sys
import numpy as np
import pandas as pd
from pyspark import SparkContext



def train_for_logreg(feature_data,labels):
    pca = PCA(n_components=50)
    principal_components = pca.fit_transform(feature_data)
    classifier = LogisticRegression(input=principal_components[:principal_components.shape[0]//2], label=output_labels[:output_labels.shape[0]//2])
    
    learning_rate = 0.00000000002

    sc = SparkContext('local[*]', 'RANDOM')
    for epoch in xrange(101):
        classifier.train_parallel(scon=sc,lr=learning_rate)
        #classifier.train(lr=learning_rate)
        cost = classifier.negative_log_likelihood()
        print('Training epoch %d, cost is ' % epoch, cost)    
        learning_rate*=0.99
   
    output_ans = labels.argmax(axis=1)
    predict_ans = classifier.predict(principal_components[:principal_components.shape[0]//4]) 
    predict_ans = predict_ans.argmax(axis=1)
    print("ACCURACY IS ",(predict_ans ==output_ans[:output_ans.shape[0]//4]).sum()/predict_ans.shape[0])
    



if __name__ == "__main__":
    feature_data = pd.read_csv("pred_feature_data.csv",sep=",").values[:,1:]
    """start here  

    feature_data = pd.read_csv("new_feature_short.csv",sep=",",header=None).values
    df = pd.read_csv("500_rock" + ".csv",sep=",").values
    if(df.shape[1]>2):
        df = df[:,1:]
    
    X = feature_data[feature_data[:,0]==df[0][0]][:,1:]
    for item in df[1:]:
        temp_data =  feature_data[feature_data[:,0]==item[0]][:,1:]
        
        #print(temp_data.shape)
        X = np.vstack((X,temp_data))

    feature_data = X
    end here"""

    output_labels = pd.read_csv("genre_label.csv",sep=",").values[:,1:]
    print(output_labels.shape)
    print(feature_data.shape)
    train_for_logreg(feature_data,output_labels)
    
    
