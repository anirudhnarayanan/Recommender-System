#!/usr/bin/env python

#Understanding User patterns using his past ratings

from ml import LinearRegress,LMS,LogisticRegression
import pandas as pd
import numpy as np
from pyspark import SparkContext
from sklearn.preprocessing import LabelBinarizer
from past.builtins import xrange


sc = SparkContext('local[*]', 'RANDOM')

def understand_user(username,feature_data,track_data):
    df = pd.read_csv(username + ".csv",sep=",").values
    #vertical_data = []
    #print(df)
    if(df.shape[1]>2):
        df = df[:,1:]
    
    X = feature_data[feature_data[:,0]==df[0][0]][:,1:]
    for item in df[1:]:
        #print(item[0])
        #print(item[0].shape)
        #print(item[1])
        #print(feature_data[feature_data[:,0]==item[0]][:,1:].shape)
        temp_data =  feature_data[feature_data[:,0]==item[0]][:,1:]
        
        #print(temp_data.shape)
        X = np.vstack((X,temp_data))
        
    
    #print("XSHAPE",X.shape)
    print(df[:,1])
    #df[:,1] =  9
    print(df[:,1])
    T = np.asmatrix(df[:,1]).T
    lb = LabelBinarizer()
    testdf = lb.fit_transform(T).tolist()
    testdf = np.array(testdf)
    print(type(testdf),testdf.shape,testdf)
    classifier = LogisticRegression(input=X, label=testdf)

    learning_rate = 0.000000008
    for epoch in xrange(100):
        classifier.train(lr=learning_rate)
        cost = classifier.negative_log_likelihood()
        print('Training epoch %d, cost is ' % epoch, cost)
        learning_rate*=0.93
    

    ans = classifier.predict(X)
    print(ans)
    ans = ans.argmax(axis=1)
    #outl = T.argmax(axis=1)
    outl = df[:,1]
    print(outl.shape)
    print(ans.shape,ans)
    print(outl)
    print(ans==outl)
    print((ans==outl).sum())
    print((ans==outl).sum()/ans.shape[0])
   
    
    

#picking random and using model on

def recommend(linreg,feature_data,track_data):
    X = feature_data[100:200,0:]
    X2 = track_data[100:200,0:][:,track_data.shape[1]-3:track_data.shape[1]-1]
    useout = linreg.use(X[:,1:])
    useout[useout>10]=10

    useout[useout<0]=0
    #print(np.asmatrix(X[:,0]).T.shape)
    track_rec = np.hstack((X2,np.asmatrix(X[:,0]).T,useout))
    track_rec = np.array(track_rec)
    print(track_rec)
    track_rec = track_rec[track_rec[:,3].argsort()][::-1]
    return track_rec



def get_recommendation(username,feature_data,track_data):
    linreg = understand_user(username,feature_data,track_data)
    val = recommend(linreg,feature_data,track_data)
    print("---------------\n")
    print("HERE ARE YOUR RECOMMENDATIONS")
    return val[:25,:]
    



if __name__ == "__main__":
    feature_data = pd.read_csv("new_feature_short.csv",sep=",",header=None).values
    print(np.asmatrix(feature_data[:,0]).T.shape)
    print(feature_data[:,feature_data.shape[1]-25:feature_data.shape[1]-1].shape)
    #feature_data = np.hstack((np.asmatrix(feature_data[:,0]).T,feature_data[:,feature_data.shape[1]-25:feature_data.shape[1]-1]))

    #feature_data = np.hstack((np.asmatrix(feature_data[:,0]).T,feature_data[:,feature_data.shape[1]-23:feature_data.shape[1]]))
    track_data = pd.read_csv("new_track_short.csv",sep=",",header=None).values
    
    print(get_recommendation("200_rock",feature_data,track_data))
    #print(understand_user("anirudh",feature_data,track_data))
    print("\n---------")

