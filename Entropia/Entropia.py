# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:33:59 2019

@author: Mario
"""
import numpy as np
import pandas as pd
from scipy.io import arff, loadmat
from math import log


def freq(x):
    y= np.unique(x)
    population_y = dict.fromkeys(y,0) # Init dictionary with data's classes
    sum_Y = len(x)
    for i in range(0,sum_Y):
        population_y[x[i]] +=1 # Count population of each class    
#    print(population_y)
    px = dict.fromkeys(population_y,0)# Init dictionary with same classes
    for key in population_y.keys():
        px[key] = population_y[key]/sum_Y #Calculate procentage of each class
    return [population_y,px]
    
    
def freq2(x,y):
    population_table = dict.fromkeys(x,0)    
    for key in population_table.keys():
        X_Y = dict.fromkeys(y,0)
        population_table[key] = X_Y
        # Count population of each feuture
    for j in range(0,x.shape[0]):                                               
        population_table[x[j]][y[j]] += 1
        # Count probability of each feuture based on their population
    population_table_probability = dict.fromkeys(x,0) 
    for key in population_table_probability.keys():
        X_Y = dict.fromkeys(y,0)
        population_table_probability[key] = X_Y
    for j in population_table:
        pop_sum = 0
        for i in dict.fromkeys(y,0):
            pop_sum += population_table[j][i]        
        for i in dict.fromkeys(y,0):
            item = population_table[j][i]
            population_table_probability[j][i] = (item/pop_sum)
        
    return [population_table,population_table_probability]
    

def entropy(Y,X = []):
    if len(X) == 0:
         # H(Y)
        entropia = 0
        py = freq(Y)[1]        
        for key in py.keys():            
            if py[key] != 0:
                entropia += py[key] * log(py[key],2)
        entropia *= -1
        return entropia
    else:        
        pyx = freq2(X,Y)
        
        px = pyx[0]
        pyx = pyx[1]
        entropie_warunkowe = []
        
        for val in pyx.values():            
            H = 0
            
            for k in val:
                if val[k] != 0:                    
                    H += val[k]*log(val[k],2)
            entropie_warunkowe.append(H*-1)         
        Hx = []
        for val in px.values():
            suma = 0
            for v in val:
                suma += val[v]                
            Hx.append(suma)
       
        entropia = 0
        for i in range(0,len(entropie_warunkowe)):            
            entropia += entropie_warunkowe[i] * (Hx[i]/sum(Hx))
        return entropia



def info_gain(X,Y):
    return entropy(Y) - entropy(Y,X)

def select_atr(X,Y,n):
    atr = dict()
    for i in range(0,X.shape[1]):
       atr[i] = (info_gain(d,X[:,i]))    
    
    atr = sorted(atr.items(), key =lambda kv:(kv[1], kv[0]),reverse = True)
    return [atr[0:n],atr]
    

def load_data(name):
    # READ AND PREPARE DATA FROM FILE
    if name.split('.')[1] == 'mat':
        #DATA FROM .mat FILES
        reuters = loadmat(name)
        names = reuters['TOPICS_COLUMN_NAMES'][0] #COLUMNS NAMES       
        X= reuters['TOPICS'] #TRAITS
        d= reuters['PLACES'][:,67]  #CLASSES      
    else:
        #DATA FROM .arff FILES
        data = arff.loadarff(name)
        df = pd.DataFrame(data[0])    
        names = df.columns #COLUMNS NAMES
        X = np.array(df)
        X = X.astype('U13')
        d = X[:,-1]   #CLASSES
        X = X[:,0:-1] #TRAITS      
    
    return [X,d,names]
        
if __name__ == "__main__":
    files = ['contact-lenses.arff','zoo.arff','reuters.mat']
    n = [2,7,20] #NUMBERS OF TRAITS TO CHOOSE
    for p in range(0,len(files)):
        print('###############')
        print('Plik',files[p].split('.')[0])
        [X,d,names] = load_data(files[p]) #GET DATA            
        atr = select_atr(X,d,n[p])
        atrn = atr[0]
        atr = atr[1]
        
        #PRINT RESULTS
        for i in range(0,len(atr)):            
            print('id:',atr[i][0],' name: ',names[atr[i][0]],' Info Gain: ',round(atr[i][1],6))
        print('\nWybrane zmienne dla pliku: ',files[p].split('.')[0])
        for i in range(0,len(atrn)):
            print(atrn[i][0],names[atr[i][0]])
        
