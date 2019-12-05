#!/usr/bin/env python

#Understanding User patterns using his past ratings

from ml import LinearRegress,LMS
import pandas as pd
import numpy as np
from pyspark import SparkContext


sc = SparkContext('local[*]', 'RANDOM')

def understand_user(username,feature_data,track_data):
    df = pd.read_csv(username + ".csv",sep=",",header=None).values
    #vertical_data = []
    #print(df)
    X = feature_data[feature_data[:,0]==df[0][0]][:,1:]
    for item in df[1:]:
        #print(item[0])
        #print(item[0].shape)
        #print(item[1])
        #print(feature_data[feature_data[:,0]==item[0]][:,1:].shape)
        temp_data =  feature_data[feature_data[:,0]==item[0]][:,1:]
        
        print(temp_data.shape)
        X = np.vstack((X,temp_data))
        
    
    print("XSHAPE",X.shape)
    T = np.asmatrix(df[:,1]).T
    print(T)
    linreg = LinearRegress()
    #print(X.shape)
    #linreg = LMS(0.002)
    linreg.train_parallel(X,T,sc)
    print("linregweight",linreg.w)
    return linreg


#picking random and using model on

def recommend(linreg,feature_data,track_data):
    X = feature_data[100:200,0:]
    X2 = track_data[100:200,0:][:,track_data.shape[1]-2:track_data.shape[1]-1]
    useout = linreg.use(X[:,1:])
    useout[useout>10]=10
    #print(np.asmatrix(X[:,0]).T.shape)
    track_rec = np.hstack((X2,np.asmatrix(X[:,0]).T,useout))
    track_rec = np.array(track_rec)
    print(track_rec)
    track_rec = track_rec[track_rec[:,2].argsort()][::-1]
    return track_rec



def get_recommendation(username,feature_data,track_data):
    linreg = understand_user(username,feature_data,track_data)
    val = recommend(linreg,feature_data,track_data)
    print("---------------\n")
    print("HERE ARE YOUR RECOMMENDATIONS")
    return val[:5,:]
    



if __name__ == "__main__":
    feature_data = pd.read_csv("feature_short.csv",sep=",").values
    track_data = pd.read_csv("track_short.csv",sep=",").values
    
    print(get_recommendation("nisar",feature_data,track_data))
    #print(understand_user("anirudh",feature_data,track_data))
    print("\n---------")
