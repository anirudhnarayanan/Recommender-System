from __future__ import print_function

import re
import sys
from operator import add

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd


def add_ones(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def least_squares(csv):
    """Parses a urls pair string into urls pair."""
    #print("CEEYESVEE",csv,csv.shape)
    #nums = csv.split(",")
    #num1 = float(nums[0])
    #num2 = float(nums[1])
    
    #X = np.asmatrix(np.array(num1))
    #T = np.asmatrix(np.array(num2))
    #X = add_ones(X)
    T = np.asmatrix(csv)
    T = np.hstack((np.ones((T.shape[0], 1)), T))
    xtx = T.T.dot(T)
    
    #arr = ["weights",np.array([1,1*num2,num2*1,num2*num2])]
    arr = ["weights",xtx]

    return arr
   
def interval_calc(csv1,csv2):
    #nums = csv.split(",")
    #num1 = float(nums[0])
    #num2 = float(nums[1])

    #X = np.asmatrix(np.array(num1))
    #T = np.asmatrix(np.array(num2))
    X = csv1
    T = csv2
    print("X",X)
    print("T",T)
    X = np.asmatrix(X)
    T = np.asmatrix(T)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    interval = X.T.dot(T)
    arr = ["weights",interval]
    return arr
    
    

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

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    #links = lines.flatMap(lambda urls: urls.split("\n")).map(lambda word: least_squares(word)).reduceByKey(lambda a,b:a+b)
    df = pd.read_csv(sys.argv[1],sep=",",header=None)
    randf = df.values
    loop_through = spark.sparkContext.parallelize([x for x in randf[:,1]])
    links = loop_through.map(lambda word: least_squares(word)).reduceByKey(lambda a,b:a+b)
    
    #print(links.collect())
    count=0
    for link,rank in links.collect():
        count = count+1
        print(link,rank)
        xtx = rank
    print("Count",count)
    loop_through = spark.sparkContext.parallelize([ (x,y) for x,y in zip(randf[:,0],randf[:,1])])
    print(loop_through.collect())
    links2 = loop_through.map(lambda word: interval_calc(word[1],word[0])).reduceByKey(lambda a,b:a+b)
    #print(links2.collect())

    for link,rank in links2.collect():
        print(link,rank)
        interval = rank
    #print(xtx.reshape((xtx.shape[0]/2,xtx.shape[0]/2)))
    print("XTX",xtx)
    #xtx = xtx.reshape((xtx.shape[0]/2,xtx.shape[0]/2))
    xtx = np.linalg.pinv(xtx)
    
    for link,rank in links2.collect():
        interval = rank
    
    rank = np.asmatrix(rank)
    rank = rank 
    

    print("FINAL WEIGHTS = ")
    weights = xtx.dot(rank)
    print(weights)
    
    df = pd.read_csv(sys.argv[1],sep=",",header=None)
    df = np.asmatrix(df.iloc[:,1].values).T
    print(df.shape) 
    df = np.hstack((np.ones((df.shape[0],1)),df))
    #print(df.dot(weights))
    print("PREDICTIONS")
    print(df.dot(weights))
    
     
    
    
    
    
