# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 08:37:05 2020

@author: Mario
"""


import itertools
import matplotlib.pyplot as plt
import scipy.io as sio
import time

def binaryzacja(X):
    for i in range(0,len(X)):
        for j in range(0,len(X[i])):
            if(X[i][j] > 0):
                X[i][j] = 1
    return X

def supp(args,dane):
    s = 0    
    for i in range(dane.shape[0]):
        row = []
        for arg in args:
            row.append(dane[i][arg-1])        
        if all(list(map(lambda x: x == 1,row))):
            s+=1        
    return s

def wyznaczanie_regul(zbiory_czeste):
    # print('Otrzymalem zbiory: ',zbiory_czeste)
    reguly = list()
    print('================ MOZLIWE REGULY ================')
    for zbior in zbiory_czeste:
        for zbiory in zbiory_czeste:  
            
            if zbior_w_zbiorze(zbior,zbiory):            
                print_regula(zbior,zbiory)
                reguly.append( (zbior,zbiory ))
    return reguly

def print_regula(zbior1,zbior2):
    s = '{'
    for z in zbior1:
        s+='A'+str(z)+','
    s = s[:-1] + '} -> {'
    for z in zbior2:
        if z not in zbior1:
            s += 'A'+str(z)+','
    s = s[:-1] + '}'
    print(s)

def zbior_w_zbiorze(zbior1,zbior2):    
    if zbior1 == zbior2 or type(zbior1)==int:        
        return False
    for i in zbior1:
        if i not in zbior2:            
            return False
    return True

def zbiory_czeste(wsparcie,minSupp):    
    return filter(lambda x: wsparcie[x]>=minSupp,wsparcie)
      
def podzbiory(klucze,i):
    return itertools.combinations(klucze,i)

def wyznacz_zbiory(dane,minWsparcie):
    zbiory = []
    wsparcie = dict()
    keys = list(range(1,dane.shape[1]+1))
    for i in range(1,dane.shape[1]+1):
        print('====================',i,'====================')
        pdzbiory = podzbiory(dict.fromkeys(keys),i)    
        wsp = dict.fromkeys(pdzbiory,0)        
        for k in wsp.keys():
            wsp[k] = supp(k,dane)
            
        # print('Wsp przed del:\n',wsp)
        for k in wsp.copy().keys():
            # print('Element key: ',k,' value: ',wsp[k])
            if wsp[k] < minWsparcie:
                del wsp[k]
        print('Wsp:\n ',wsp)
        if wsp == {}:
            print('BRAK PODZBIOROW O ROZMIARZE ',i,'O ZAUFANIU WIEKSZYM NIZ MINIMALNE \n'
                  '=================== PRZERYWAM ===================\n')
            
            break
        wsparcie.update(wsp)
        # print('Wsparcie:\n ',wsparcie)
        zb = list(zbiory_czeste(wsp,minWsparcie))
        for z in zb:
            if z not in zbiory:
                zbiory.append(z)
            
        # print('Zb:\n ',zb)
        keys = list(dict.fromkeys(list(itertools.chain(*zb))))
        # print('Zbiory:\n ',zbiory)        
        print('Keys:\n ',keys)    
    return [zbiory,wsparcie]

def zaufanie(reguly,wsparcie):
    zauf = dict()    
    for tup in reguly:
        zauf[tup] = wsparcie[tup[1]]/wsparcie[tup[0]]        
    return zauf

def minimalne_zaufanie(zaufanie,min_zaufanie):
    return filter(lambda x: zaufanie[x]>min_zaufanie,zaufanie)

def dane_do_wykresu(ostateczne_reguly,wsparcie,zaufanie):
    wsp = []
    zauf = []
    for regula in ostateczne_reguly:
        wsp.append(wsparcie[regula[1]])
        zauf.append(zaufanie[regula])
    return [wsp,zauf]

def wykres(dane,maxX):    
    plt.plot(dane[0],dane[1],'k.')
    plt.xlim(left=0,right=maxX)
    plt.ylim(bottom=0,top=1.1)
    
def brzeg_pareto(dane):
    brzeg = dict()         
    dane = przetworz_dane(dane)      
    maxWsparcie = max(dane.keys())   
    m = 0    
    for i in range(maxWsparcie,0,-1):        
        if i in dane.keys() and dane[i]>m:
            brzeg[i] = dane[i]
            m = dane[i]
    plt.plot(list(brzeg.keys()),list(brzeg.values()),'r.')    
    keys = list(brzeg.keys())    
    for i, k  in enumerate(keys):         
        if i == 0:
            plt.plot([k,k],[0,brzeg[k]],':r')
        if i == len(brzeg.items())-1:
            plt.plot([k,0],[brzeg[k],brzeg[k]],':r')
        if i<len(brzeg.items())-1:
            plt.plot([keys[i+1],k],[brzeg[keys[i]],brzeg[keys[i]]],':r')
            plt.plot([keys[i+1],keys[i+1]],[brzeg[keys[i]],brzeg[keys[i+1]]],':r')
    return brzeg
                
def przetworz_dane(dane):    
    wynik = dict.fromkeys(dane[0],0)    
    for i in range(len(dane[0])):        
        if wynik[dane[0][i]]<dane[1][i]:
            wynik[dane[0][i]] = dane[1][i]
    return wynik


def apriori(dane,minWsparcie,minZaufanie):
    start = time.time()
    [zbiory,wsparcie] = wyznacz_zbiory(dane,minWsparcie)    
    reguly = wyznaczanie_regul(zbiory)
    # print('Reguly:\n ',reguly)
    zauf = zaufanie(reguly,wsparcie)    
    ostateczne_reguly = list(minimalne_zaufanie(zauf,minZaufanie))
    print(' ================ OSTATECZNE REGULY ================')
    for r in ostateczne_reguly:
        print_regula(r[0],r[1])
        print('Zaufanie:',round(zauf[r],4))
        print('Wsparcie: ',round(wsparcie[r[1]]))
    #wykres
    dane_wykres = dane_do_wykresu(ostateczne_reguly,wsparcie,zauf)
    
    wykres(dane_wykres,1.1*max(dane_wykres[0]))
    brzeg = brzeg_pareto(dane_wykres)
    print(' ================ REGULY NA BRZEGU ================')    
    for k,v in zauf.items():
        for b in brzeg:
            if v == brzeg[b]:
                print_regula(k[0], k[1])
                print('Zaufanie: ',brzeg[b])
                print('Wsparcie: ',b)
    
    plt.title('Wykres dla zbioru z '+str(dane.shape[1])+' atrybutami dla:\n minimalnego wsparcia: '+str(minWsparcie)+' \n minimalnego zaufania: '+str(minZaufanie))
    plt.show()
    print('================ CZAS ================')
    print("Czas wykonywania: ",round(time.time()-start,2),' s')
    print("Ilosc zbiorow istotnych: ",len(ostateczne_reguly))
    
""" Przyklad z pdf
#A1 = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
#A2 = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0]
#A3 = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]
#A4 = [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
#A = [A1,A2,A3,A4]
#A = np.transpose(np.array(A))
#apriori(A,6,0.5)
"""



reuters = sio.loadmat('reuters.mat')
reuters = reuters['TOPICS'][:1000,:]
apriori(reuters,2,0.6)
