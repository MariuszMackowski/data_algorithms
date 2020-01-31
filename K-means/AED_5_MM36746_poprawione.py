# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 08:55:48 2019

@author: Mario
"""
import numpy as np
import pandas as pd
from scipy.io import arff
import random
from statistics import mean
import matplotlib . pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import cdist
import time

def load_data(name):    
    #DATA FROM .arff FILES
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])    
    names = df.columns #COLUMNS NAMES
    X = np.array(df)
    X = X.astype('U13')
    d = X[:,-1]   #CLASSES
    X = X[:,0:-1] #TRAITS      
    X = X.astype(np.float64)
    
    return [X,d,names]


def losuj_centra(X,n):
    centra = []
    for j in range(n):
        centrum = []
        for i in range(X.shape[1]):            
            centrum.append(round(random.uniform(min(X[:,i]),max(X[:,i])),1))
        centra.append(centrum)
    return centra
    
def odleglosc_euklidesowa(X,C):    
    return round(sum((X[i]-C[i])**2 for i in range(len(C))),2)




def odleglosc_Mahalanobisa(X,C):    
    X = np.asarray(X)
    C = np.asarray(C)        
    i = C.shape[0]    
    C = C.reshape(i,1).T    
    VI = np.linalg.inv(np.cov(X.T))
    delta = X - C    
    m = np.diag(np.sqrt(np.dot(np.dot(delta, VI), delta.T)))    
    return m


def najblizsze_centrum(X,C):    
    najblizsze = dict.fromkeys(range(len(C)))
    for k in najblizsze:
        najblizsze[k] = list()   
    
    for i in range(X.shape[0]):        
        m = []
        for c in C:            
            m.append(odleglosc_euklidesowa(X[i],c))
        
        m = m.index(min(m))
        najblizsze[m].append(i)
    
    return najblizsze

def najblizsze_centrum_malacha(X,C):    
    najblizsze = dict.fromkeys(range(len(C)))
    for k in najblizsze:
        najblizsze[k] = list()  
    m = np.zeros((len(X),len(C)))
    
    for c in range(len(C)):
        m[:,c] = odleglosc_Mahalanobisa(X,C[c])
    
    for i in range(X.shape[0]):  
        naj, = np.where(m[i] == min(m[i]))
        naj = int(naj)
        najblizsze[naj].append(i)    
    return najblizsze

def nowy_srodek(X,C):
    najblizsze = najblizsze_centrum(X,C) #najblizsze_centrum(X,C) <- zamiana odleglosci
    nowe_srodki = []
    for i in range(len(najblizsze)):        
        x = X[najblizsze[i]]
        x_mean = []
        for j in range(x.shape[1]):
            if x.shape[0]>0:
                x_mean.append(round(mean(x[:,j]),2))
        nowe_srodki.append(x_mean)    
    return [nowe_srodki,najblizsze]

def roznica_centr(stare,nowe):
    for i in range(len(nowe)):
        if (len(stare) != len(nowe)) or (len(stare[i]) != len(nowe[i])):
            print('UWAGA! \nJedno lub wiecej centr [pustych]')
            return -1
    else:
        r = 0
        for i in range(len(stare)):
            for j in range(len(stare[0])):
                if stare[i][j] != None and nowe[i][j] != None:
                    r += abs(stare[i][j] - nowe[i][j])
    return r

def PCA(d,X,centra,najblizsze):
    #FROM PCA_MM36746.py
    arg = najblizsze 
    C = np.cov(np.transpose(X))
    l, p = np.linalg.eig(C)
    
    sorted_order = np.argsort(l)    
    p_sorted = p[:,sorted_order[::-1]]
    
    nowyX = np.dot(X,p_sorted)
    nie_wyswietlac = []
    j = X.shape[1]
    
    for i in range(len(centra)):
        if len(centra[i])<=0:
            centra[i] = list(map(float,list('0'*j)))
            nie_wyswietlac.append(i)
    
    centra = np.dot(centra,p_sorted)    
    fig = plt.figure()
    for clas in arg:        
        if len(arg[clas])>0:
            x = np.array(arg[clas])        
            plt.plot(nowyX[x,0],nowyX[x,1],'.')
    for c in range(len(centra)):
        if c not in nie_wyswietlac:
            plt.plot(centra[c][0],centra[c][1],'or')
        
    fig = plt.figure()
    ax = fig.gca( projection = '3d')
    for clas in arg:        
        if len(arg[clas])>0:
            x = np.array(arg[clas])                
            ax.plot(nowyX[x,0],nowyX[x,1],nowyX[x,2],'.')
    
    for c in range(len(centra)):        
        if c not in nie_wyswietlac:       
            plt.plot([centra[c][0]],[centra[c][1]],[centra[c][2]],'ro')
    plt.show()
    

def PCA_stare(X=None,d=None,name=None):
    if name!=None:
        data = arff.loadarff(name)
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
#    for i in range(0,len(l)):
#        print('Wartosc'+str(i)+': '+ str(l[i]))
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
    plt.show()        
    

def losujXY(mi,ma,ilosc_punktow,ilosc_atr,ilosc_klas):
    X = (ma-mi)*np.random.rand(ilosc_punktow,ilosc_atr)+mi

    Y = np.round((ilosc_klas-1)*np.random.rand(ilosc_punktow,1)).astype(str).ravel()
   
    return [X,Y]

if __name__ == '__main__': 
    plik = 'iris.arff'
    start = time.time()
 
    [X,Y] = load_data(plik)[0:2]
    [X,Y] = losujXY(-10,10,100,3,2)
    ilosc_centr = 2
    nowe_centra = losuj_centra(X,ilosc_centr)

    roznica = 1
    warunek_stopu = 0.01
    while roznica>warunek_stopu:
        stare_centra = nowe_centra
        [nowe_centra,najblizsze] = nowy_srodek(X,nowe_centra)
        roznica = roznica_centr(stare_centra,nowe_centra)
    PCA(Y,X,nowe_centra,najblizsze)
    
    end = time.time()
    elapsed_time = end - start
    print(end-start)
   
    PCA_stare(X,Y)
#plik = 'iris.arff'
#[X,Y] = load_data(plik)[0:2]
#ilosc_centr = 3
#nowe_centra = losuj_centra(X,ilosc_centr)
#
#odl = odleglosc_Mahalanobisa(X,nowe_centra[0])
#print(odl)
#print(odl.shape)
