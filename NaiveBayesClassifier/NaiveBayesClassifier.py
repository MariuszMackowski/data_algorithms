"""
Created on Wed Oct 16 22:15:44 2019

@author: Mario
"""

import numpy as np
import pandas as pd
from scipy.io import arff
import operator
import matplotlib.pyplot as plt

class naiveBayes:
    """
    Naive Bayes is a simple technique for constructing classifiers: 
    models that assign class labels to problem instances,
    represented as vectors of feature values,
    where the class labels are drawn from some finite set.
    """
    
    def __init__(self, clas, data, laplace):
        """
        Initiate classifier with data and set laplace flag
        """
        self.clas = clas
        self.data = data 
        if laplace:
            self.laplace = 1
        else:
            self.laplace = 0     

    def fit(self):
        """
        Method creates tables of classes populations, classes probability, feutures population and feutures probability.
        """
        # Class population and probability
        y= np.unique(self.clas)
        self.population_y = dict.fromkeys(y,self.laplace) # Init dictionary with data's classes
        self.sum_Y = len(self.clas)
        for i in range(0,self.sum_Y):
            self.population_y[self.clas[i]] +=1 # Count population of each class        
        self.px = dict.fromkeys(self.population_y,0)# Init dictionary with same classes
        for key in self.population_y.keys():
            self.px[key] = self.population_y[key]/self.sum_Y #Calculate procentage of each class
            
        # Feutures population and probability
        # Both tables are 3D matrix, so I used dictionary with 3 keys [row][class][trait]
        #Init both dictionaries
        self.population_table = dict.fromkeys(range(1,self.data.shape[1]+1),self.laplace) 
        self.population_table_probability = dict.fromkeys(range(1,self.data.shape[1]+1),self.laplace)
        for i in range(1,self.data.shape[1]+1):
            inner_table = dict.fromkeys(self.px,0)            
            self.population_table[i] = inner_table
            self.population_table_probability[i] = inner_table
            for key in self.population_table[i].keys():
                self.X_Y = {'false':self.laplace,'true':self.laplace}
                self.population_table[i][key] = self.X_Y  
        # Count population of each feuture
        for i in range(1,self.data.shape[1]+1):            
            for j in range(0,self.data.shape[0]):                                               
                self.population_table[i][self.clas[j]][self.data[j,i-1]] += 1
        # Count probability of each feuture based on their population
        for i in range(1,len(self.population_table)+1):
            for j in self.population_y:
                pop_sum = self.population_table[i][j]['true'] + self.population_table[i][j]['false']
                self.population_table_probability[i][j]['true'] = self.population_table[i][j]['true']/pop_sum
                self.population_table_probability[i][j]['false'] = self.population_table[i][j]['false']/pop_sum

    def predict(self,x):
        """
        Method classifys sample to one of learned classes, based on the highest probability. 
        It returns class name.
        """
        p = dict.fromkeys(self.population_y,1.0) # Get all learned classes     
        for c in p.keys():
            for i in range(1,len(x)+1):
                p[c] *= self.population_table_probability[i][c][x[i-1]] # Calculate probability for each class
        m = max(p.items(), key=operator.itemgetter(1))[0]# Find maximum
        return m
        
    
def split_data(x,d,alfa):
    """
    Method splits data for learning and testing sets. Alfa varable is sets divide ratio.
    """
    idx = np.random.permutation(len(x)) # Shuffle whole set
    alfa = round(len(x)* alfa) # Calculate lengths of sets 
    xu = x[idx[0:alfa],:] # Split
    du = d[idx[0:alfa]]
    xt = x[idx[alfa:len(x)+1],:]
    dt = d[idx[alfa:len(x)+1]]
    return xu,du,xt,dt

def calculateAcc(y,y_pred):
    """
    Method caluclate accuracy of classification result on test set.
    """
    acc = 0
    for i in range(0,len(y)):
        if y[i] == y_pred[i]: #If classificator is right, increase score
           acc+=1
    acc /= len(y) # Calculate procentage of right classications
    return acc      
  
        
if __name__ == "__main__":
    # READ AND PREPARE DATA FROM FILE
    data = arff.loadarff('zoo.arff')
    df = pd.DataFrame(data[0])
    X = np.array(df)
    X = X.astype('U13')
    idx = []
    for i in range(1, len(data[1].types())-1):
        if data[1].types()[i] == 'nominal':
            idx.append(i)            
    # SPLIT CLASSES AND TRAITS
    d = X[:,-1]
    X = X[:,idx]
    
    # CALCULATE ACCURACY OF CLASSIFACATIONS FOR DIFFRENT ALFA RATIO AND 100 TRIES FOR EACH
    no_prob = 100
    alfa = np.arange(0.05,0.95,0.05)
    acc = []
    acc_with_correction = []
    # CLASSIFIER WITHOUT LAPLACE CORRECTION
    for i in range(len(alfa)):
        sum_acc = 0.0
        nB = None
        #COUNT ACCURACY FOR EACH ALFA
        for j in range(no_prob):
            [xu,du,xt,dt] = split_data(X,d,float(alfa[i]))
            nB = naiveBayes(du,xu,False)
            nB.fit()
            classicitation_result = []
            for k in range(0,len(xt)):
                classicitation_result += [nB.predict(xt[k])]
            sum_acc += calculateAcc(classicitation_result, dt)
        sum_acc /=no_prob
        acc.append(sum_acc)
    # CLASSIFIER WITH LAPLACE CORRECTION        
    for i in range(len(alfa)):
        sum_acc = 0.0
        nB = None
        #COUNT ACCURACY FOR EACH ALFA
        for j in range(no_prob):
            [xu,du,xt,dt] = split_data(X,d,float(alfa[i]))
            nB = naiveBayes(du,xu,True)
            nB.fit()
            classicitation_result = []
            for k in range(0,len(xt)):
                classicitation_result += [nB.predict(xt[k])]
            sum_acc += calculateAcc(classicitation_result, dt)
        sum_acc /=no_prob    
        acc_with_correction.append(sum_acc)
    # PLOT RESULTS
    plt.plot(alfa,acc,'b',alfa,acc_with_correction,'r')
    plt.legend(['Results without Laplace\'s correction','Results with Laplace\'s correction'])
    plt.xlabel('Alfa')
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes Classifier Accuracy')
    plt.grid()
    plt.show()
#    

