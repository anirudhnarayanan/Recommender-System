#!/usr/bin/env python

import numpy as np 
from abc import ABC, abstractmethod

# Super class for machine learning models 

class BaseModel(ABC):
    """ Super class for ITCS Machine Learning Class"""
    
    @abstractmethod
    def train(self, X, T):
        pass

    @abstractmethod
    def use(self, X):
        pass

    
class LinearModel(BaseModel):
    """
        Abstract class for a linear model 
        
        Attributes
        ==========
        w       ndarray
                weight vector/matrix
    """

    def __init__(self):
        """
            weight vector w is initialized as None
        """
        self.w = None

    # check if the matrix is 2-dimensional. if not, raise an exception    
    def _check_matrix(self, mat, name):
        if len(mat.shape) != 2:
            raise ValueError(''.join(["Wrong matrix ", name]))
        
    # add a basis
    def add_ones(self, X):
        """
            add a column basis to X input matrix
        """
        self._check_matrix(X, 'X')
        return np.hstack((np.ones((X.shape[0], 1)), X))

    ####################################################
    #### abstract funcitons ############################
    @abstractmethod
    def train(self, X, T):
        """
            train linear model
            
            parameters
            -----------
            X     2d array
                  input data
            T     2d array
                  target labels
        """        
        pass
    
    @abstractmethod
    def use(self, X):
        """
            apply the learned model to input X
            
            parameters
            ----------
            X     2d array
                  input data
            
        """        
        pass 


# Linear Regression Class for least squares
class LinearRegress(LinearModel): 
    """ 
        LinearRegress class 
        
        attributes
        ===========
        w    nd.array  (column vector/matrix)
             weights
    """
    def __init__(self):
        LinearModel.__init__(self)

    def least_squares(self,csv):
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
       
    def interval_calc(self,csv1,csv2):
        #nums = csv.split(",")
        #num1 = float(nums[0])
        #num2 = float(nums[1])

        #X = np.asmatrix(np.array(num1))
        #T = np.asmatrix(np.array(num2))
        X = csv1
        T = csv2
        #print("X",X)
        #print("T",T)
        X = np.asmatrix(X)
        T = np.asmatrix(T)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        interval = X.T.dot(T)
        arr = ["weights",interval]
        return arr
        
    # train lease-squares model
    def train(self, X, T):
        X = self.add_ones(X)
        print("xshape",X.shape)
        print("Tshape",T.shape)
        xtx = X.T.dot(X)
        #print(xtx)
        print("xtx shape",xtx.shape)
        self.w = np.linalg.pinv(xtx)
        print(X.T.shape)
        print(T.shape)
        interval = (X.T).dot(T)
        print("interval shape",interval.shape)
        self.w = self.w.dot(interval)
        print("w shape",self.w.shape)
        self.w = self.w.T
        print(self.w)
        return self.w.T
        ## TODO: replace this with your codes
    
    def train_parallel(self, X, T,sc):
        #X = self.add_ones(X)
        print("xshape",X.shape)
        print("Tshape",T.shape)
        
        loop_through = sc.parallelize([x for x in X])
        links = loop_through.map(lambda word: self.least_squares(word))#.reduceByKey(lambda a,b:a+b)
        print(links.collect())
        for link,rank in links.collect():
            print(link,rank)
            xtx = rank
        
        loop_through = sc.parallelize([ (x,y) for x,y in zip(X,T)])
        links2 = loop_through.map(lambda word: self.interval_calc(word[0],word[1])).reduceByKey(lambda a,b:a+b)

        for link,rank in links2.collect():
            interval = rank

        xtx = np.linalg.pinv(xtx)

        for link,rank in links2.collect():
            interval = rank

        rank = np.asmatrix(rank)
        rank = rank


        print("FINAL WEIGHTS = ")
        weights = xtx.dot(rank)
        self.w = weights
        return self.w

    # apply the learned model to data X
    def use(self, X):
        X = self.add_ones(X)
        hypothesis = self.w.T.dot(X.T)
        return hypothesis.T
        ## TODO: replace this with your codes


import collections # for checking iterable instance

# LMS class 
class LMS(LinearModel):
    """
        Lease Mean Squares. online learning algorithm
    
        attributes
        ==========
        w        nd.array
                 weight matrix
        alpha    float
                 learning rate
    """
    def __init__(self, alpha):
        LinearModel.__init__(self)
        self.alpha = alpha
        
    # batch training by using train_step function
    def train(self, X, T):
        iterations = 300
        X = self.add_ones(X)
        self.w = np.zeros((1,X.shape[1]))
        #print("X is ")
        #print(X)
        #print("W is")
        #print(self.w)
        for i in range(iterations):            
            temp = self.w.dot(X.T)
            error = temp.T - T
            new_x = error.T.dot(X)
            self.w = self.w - ((self.alpha)*new_x)
        print("train response")
        return self.w
        pass  ## TODO: replace this with your codes
            
    # train LMS model one step 
    # here the x is 1d vector
    def train_step(self,x,t):
        x = np.insert(x,0,1)
        x = x.reshape(1,len(x))
        #print(x)
        self.w = np.zeros((1,x.shape[1]))
        #print(self.w)
        #self.w = np.zeros((1,x.shape[1]))
        mult_x = x
        error = self.w.dot(mult_x.T) - t
        #print(error)
                                                 
        update_value = ((self.alpha)*error)*x
        self.w = self.w - update_value
        print(self.w)
        return self.w
        
        
    def train_step_full(self, x, t):
        x = self.add_ones(x)
        self.w = np.zeros((1,x.shape[1]))
        print(x[0])
        for i in range(x.shape[1]):
            #print(self.w)
            mult_x = np.reshape(x[i],(len(x[i]),1))
            error = self.w.dot(mult_x) - t[i]
            update_value = ((self.alpha)*error)*x[i]
            self.w = self.w - update_value
            #print(self.w)
        print("returning stochastic")
        return self.w
        
        pass  ## TODO: replace this with your codes
    
    # apply the current model to data X
    def use(self, X):
        X = self.add_ones(X)
        hypothesis = self.w.dot(X.T)
        return hypothesis.T
        pass  ## TODO: replace this with your codes
        
import numpy as np

def sigmoid_function(x):
    return 1/(1+np.exp(-x))

def softmax_function(scores):
    print("scoreshape",scores.shape)
    exp=np.exp(scores-np.max(scores))
    if exp.ndim==1:
        return exp/np.sum(exp,axis=0)
    else:  
        return exp/np.array([np.sum(exp,axis=1)]).T 


class LogisticRegression(object):
    def __init__(self, input, label):
        self.x = input
        self.y = label
        self.W = np.zeros((self.x.shape[1], self.y.shape[1]))
        self.b = np.zeros(self.y.shape[1])

    def least_squares(self,csv):
        """Parses a urls pair string into urls pair."""
        #print("CEEYESVEE",csv,csv.shape)
        #nums = csv.split(",")
        #num1 = float(nums[0])
        #num2 = float(nums[1])
        
        #X = np.asmatrix(np.array(num1))
        #T = np.asmatrix(np.array(num2))
        #X = add_ones(X)
        T = np.asmatrix(csv)
        #T = np.hstack((np.ones((T.shape[0], 1)), T))
        xtx = T.T.dot(T)
        
        #arr = ["weights",np.array([1,1*num2,num2*1,num2*num2])]
        arr = ["weights",xtx]

        return arr
       
    def interval_calc(self,csv1,csv2):
        #nums = csv.split(",")
        #num1 = float(nums[0])
        #num2 = float(nums[1])

        #X = np.asmatrix(np.array(num1))
        #T = np.asmatrix(np.array(num2))
        X = csv1
        T = csv2
        #print("X",X)
        #print("T",T)
        X = np.asmatrix(X).T
        #X = np.hstack((np.ones((X.shape[0], 1)), X))
        interval = X.T.dot(T)
        arr = ["weights",interval]
        #return np.array(arr)
        return np.squeeze(np.array(arr))

    def train_parallel(self, scon,lr=0.1, L2_reg=0.00):
        print(self.x.shape)
        print(self.W.shape)
        print(self.b.shape)
        print(self.y.shape)
        lamb = 0.00000004
        dot_hypo = np.dot(self.x, self.W) 
        loop_through = scon.parallelize([ x for x in self.x])
        links2 = loop_through.map(lambda word: self.interval_calc(word,self.W)).reduceByKey(lambda a,b:np.vstack((a,b)))
        for link,rank in links2.collect():
            print(link,rank)
            interval = rank

        #dot_hypo = np.asmatrix(interval)
        dot_hypo = np.array(interval)
        print(dot_hypo)
        print("dothyposhape",dot_hypo.shape)
        print(type(dot_hypo))
        print(type(dot_hypo[0]))
        print(self.b.shape)
        
        
        p_y_given_x = softmax_function(dot_hypo + self.b)
        d_y = self.y - p_y_given_x
        self.W += lr * np.dot(self.x.T, d_y) - lamb * L2_reg * self.W
        self.b += lr * np.mean(d_y, axis=0)
   
    def train(self, lr=0.1, L2_reg=0.00):
        print(self.x.shape)
        print(self.W.shape)
        print(self.b.shape)
        print(self.y.shape)
        lamb = 0.0000000004
 
        dot_hypo = np.dot(self.x, self.W) 
        print("dothyposhape",dot_hypo.shape)
        print(dot_hypo)
        print(type(dot_hypo))
        print(type(dot_hypo[0]))
        print(self.b.shape)
        p_y_given_x = softmax_function(dot_hypo+ self.b)
        d_y = self.y - p_y_given_x
        self.W += lr * np.dot(self.x.T, d_y) - lr* L2_reg * self.W
        self.b += lr * np.mean(d_y, axis=0)

    def negative_log_likelihood(self):
        sigmoid_activation = softmax_function(np.dot(self.x, self.W) + self.b)
        cross_entropy = - np.mean(np.sum(self.y * np.log(sigmoid_activation) + (1 - self.y) * np.log(1 - sigmoid_activation),axis=1))

        return cross_entropy


    def predict(self, x):
        return softmax_function(np.dot(x, self.W) + self.b)
