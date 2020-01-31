# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:30:25 2019

@author: mackomar
"""
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit 
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from timeit import default_timer as timer

def load_data(name):
    # READ AND PREPARE DATA FROM FILE    
    #DATA FROM .arff FILES
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])    
    names = df.columns #COLUMNS NAMES
    X = np.array(df)
    X = X.astype(np.float)
    d = X[:,-1]   #CLASSES
    X = X[:,0:-1] #TRAITS   
    return [X,d,names]

if __name__ == "__main__":
    [X,d,names] = load_data('diabetes.arff')

    indicators = dict.fromkeys(['Accuracy','Recall','Specification','Precision',
                 'F1', 'Balanced_accuracy','AUC'],0.0) #Liczone wskazniki
    no_sh = 50 # liczba powtorzen
    classifiers = [GaussianNB(),
                   QuadraticDiscriminantAnalysis(),
                   AdaBoostClassifier(),
                   LinearDiscriminantAnalysis()] #Wybrane klasyfikatory
    
    for cla in classifiers:
        rs = ShuffleSplit(no_sh , .5, .5, 0)
        name = str(cla).split('(')[0] #Uzyskanie nazwy klasyfikatora
        print('\nResults for classifier:',name)       

        start = timer() #Poczatek zliczania czasu nauki i predykcji
        
        for train_index , test_index in rs.split(X):
           
            
            cla.fit(X[train_index],d[train_index])#nauka klasyfikatora       
            Y = cla.predict(X[test_index])#predykcja
            matrix = confusion_matrix(d[test_index],Y,labels = [0,1])#Obliczanie macierzy konfuzji
            [tn, fp, fn, tp] = matrix.ravel()#Przypisanie wartosci TN,FP,FN,TP
            
            #Liczenie wskaznikow ze wzorow i sumowanie wynikow z poprzzednimi iteracjami
            indicators['Accuracy'] += (tn+tp)/(sum(sum(matrix)))
            indicators['Recall'] += tp/(tp+fn)
            indicators['Specification'] += tn/(tn+fp)
            indicators['Precision'] += tp/(tp+fp)
            indicators['F1'] += (2 * (tp/(tp+fn)) * (tp/(tp+fp)))/((tp/(tp+fn)) + (tp/(tp+fp)))
            indicators['Balanced_accuracy'] += ( (tn/(tn+fp)) + (tp/(tp+fn)))/2
            y = cla.predict_proba(X[test_index])
            indicators['AUC'] += roc_auc_score(d[test_index],y[:,1])
            
        end = timer() #Koniec zliczania czasu nauki i predykcji
        print('Sredni czas nauki: ',round((end - start)/no_sh,4),'s')
        for k in indicators.keys(): #Usrednienie wskaznikow i ich wyswietlenie
            indicators[k] /= no_sh
            print(k,': ',round(indicators[k],5))
            
        #Liczenie wartosci krzywej ROC
        cla.fit(X[train_index],d[train_index])
        y = cla.predict_proba(X[test_index])
        roc = roc_curve(d[test_index],y[:,1])
        
        #Liczenie odleglosci między punktami krzywej a (0,1) i szukanie minimum
        odl = []        
        for i in range(len(roc[0])):
            o = np.sqrt(roc[0][i]**2 + (1-roc[1][i])**2)
            odl.append(o)
        idx = odl.index(min(odl))
        #Wyswietlanie
        plt.plot(roc[0],roc[1], label = 'ROC curve')#Krzywa ROC        
        plt.plot(roc[0][idx],roc[1][idx],'ro',label = 'Closest to (0,1)')#Punkt najblizej (0,1)
        #Przerywane linie zaznaczające minimum
        plt.plot([-0.1,roc[0][idx]],[roc[1][idx],roc[1][idx]],'r:')
        plt.plot([roc[0][idx],roc[0][idx]],[-0.05,roc[1][idx]],'r:')
        #Tekst minimum
        plt.figtext(0.2,roc[1][idx]/1.12,str(round(roc[1][idx],5)),color='red')
        plt.figtext(roc[0][idx]*1.43,0.4,str(round(roc[0][idx],5)),rotation=270,color='red')
        
        plt.grid()#Siatka
        plt.title(name)#Tytul = nazwa klasyfikatora
        plt.xlim((-0.1,1.1))#Przedzial wartosci na osi X
        plt.ylim((-0.05,1.1))#Przedzial wartosci na osi Y
        plt.legend(loc = 'right')#legenda
        plt.show()
        
        
        
