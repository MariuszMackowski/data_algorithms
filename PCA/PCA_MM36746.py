# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:37:49 2019

@author: Mario
"""
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib . pyplot as plt
from mpl_toolkits . mplot3d import Axes3D
import random



if __name__ == "__main__":
    # READ AND PREPARE DATA FROM FILE
#    data = arff.loadarff('waveform5000.arff')
    data = arff.loadarff('iris.arff')
    df = pd.DataFrame(data[0])
    X = np.array(df)
    X = X.astype('U13')

    d = X[:,-1]
    X = X[:,0:-1].astype(np.float)
    
    arg = dict.fromkeys(d,list())
    for clas in arg:
        a = []
        for i in range(0,len(d)):            
            if d[i] == clas:
                a.append(i)
        arg[clas] = a

    
    C = np.cov(np.transpose(X))
    l, p = np.linalg.eig(C)
    for i in range(0,len(l)):
        print('Wartosc'+str(i)+': '+ str(l[i]))
    sorted_order = np.argsort(l)    
    p_sorted = p[:,sorted_order[::-1]]
    
    nowyX = np.dot(X,p_sorted)
    
    
    fig = plt.figure()
    for clas in arg:        
        x = np.array(arg[clas])        
        plt.plot(nowyX[x,0],nowyX[x,1],'.')
        
    fig = plt.figure()
    ax = fig.gca( projection = '3d')
    for clas in arg:
        x = np.array(arg[clas])            
        ax.plot(nowyX[x,0],nowyX[x,1],nowyX[x,2],'.')
    

    rnd = list(range(0,X.shape[1]))
    random.shuffle(rnd)
    
            
    fig = plt.figure()
    for clas in arg:
        x = np.array(arg[clas]) 
        plt.plot(X[x,rnd[0]],X[x,rnd[1]],'.')
        plt.title('Wczytane dane na wymiarach: '+str(rnd[0])+'\\'+str(rnd[1]))
        
    random.shuffle(rnd)
    
            
    fig = plt.figure()
    ax = fig.gca( projection = '3d')
    for clas in arg:
        x = np.array(arg[clas])
        ax.plot(nowyX[x,rnd[0]],nowyX[x,rnd[1]],nowyX[x,rnd[2]],'.')
        plt.title('Wczytane dane na wymiarach: '+str(rnd[0])+'\\'+str(rnd[1])+'\\'+str(rnd[2]))
    plt.show()    