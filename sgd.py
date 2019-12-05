from __future__ import print_function

import re
import sys
from operator import add

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

alpha = 0.002

def add_ones(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

def train(X, T):
    iterations = 3000
    X = add_ones(X)
    w = np.zeros((1,X.shape[1]))
    #print("X is ")
    #print(X)
    #print("W is")
    #print(w)
    for i in range(iterations):            
	temp = w.dot(X.T)
	error = temp.T - T
	new_x = error.T.dot(X)
	w = w - ((alpha)*new_x)
    print("train response")
    return w
   
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: pagerank <file>", file=sys.stderr)
        sys.exit(-1)

    print("WARN: This is a naive implementation of PageRank and is given as an example!\n" +
          "Please refer to PageRank implementation provided by graphx",
          file=sys.stderr)

    spark = SparkSession\
        .builder\
        .appName("PythonPageRank")\
        .getOrCreate()

        
    randf = pd.read_csv("yxlin2.csv",sep=",",header=None)
    X =randf.iloc[:,0].values
    T = randf.iloc[:,1].values
    X = np.asmatrix(X)

    X = X.T
    T = np.asmatrix(T)
    T = T.T
    weights = train(T,X)
    print("FINAL WEIGHTS = ")
    print(weights)
    
    df = pd.read_csv(sys.argv[1],sep=",",header=None)
    df = np.asmatrix(df.iloc[:,1].values).T
    print(df.shape) 
    df = np.hstack((np.ones((df.shape[0],1)),df))
    print("PREDICTIONS")
    print(df.dot(weights.T))
