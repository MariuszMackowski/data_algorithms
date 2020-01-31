# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 08:37:05 2020

@author: Mario
"""


import itertools
import matplotlib.pyplot as plt
import scipy.io as sio
import time


def support(args,data):
    """   
    TO UPGRADE 
   
    """
    sup = 0    
    for i in range(data.shape[0]):
        row = []
        for arg in args:
            row.append(data[i][arg-1])        
        if all(list(map(lambda x: x == 1,row))):
            sup+=1        
    return sup

def determine_rules(frequent_sets):  
    rules = list()
    print('================ POSSIBLE RULES ================')
    for sett in frequent_sets:
        for sets in frequent_sets:            
            if set_in_set(sett,sets):            
                print_rule(sett,sets)
                rules.append( (sett,sets) )
    return rules

def print_rule(set1,set2):
    s = '{'
    for z in set1:
        s+=str(z)+','
    s = s[:-1] + '} -> {'
    for z in set2:
        if z not in set1:
            s += str(z)+','
    s = s[:-1] + '}'
    print(s)

def set_in_set(set1,set2):    
    if set1 == set2 or type(set1)==int:        
        return False
    for i in set1:
        if i not in set2:            
            return False
    return True

def frequent_sets(support,minSupp):    
    return filter(lambda x: support[x]>=minSupp,support)
      
def subsets(keys,size):
    return itertools.combinations(keys,size)

def determine_sets(data,minSupport):
    sets = []
    supp = dict()
    keys = list(range(1,data.shape[1]+1))
    for i in range(1,data.shape[1]+1):
        print('==================== SIZE ',i,' ====================')
        sbsets = subsets(dict.fromkeys(keys),i)    
        su = dict.fromkeys(sbsets,0)
        for k in su.keys():
            su[k] = support(k,data)            
        
        for k in su.copy().keys():            
            if su[k] < minSupport:
                del su[k]
        print('Frequent sets:\n ',su)
        if su == {}:
            print('NO SUBSET OF SIZE ',i,'WITH SUPPORT HIGHER THEN MINIMAL \n'
                  '=================== BREAKING ===================\n')            
            break
        supp.update(su)        
        fs = list(frequent_sets(su,minSupport))
        for z in fs:
            if z not in sets:
                sets.append(z)            
        keys = list(dict.fromkeys(list(itertools.chain(*fs))))            
    return [sets,supp]

def confidence(rules,support):
    conf = dict()    
    for tup in rules:
        conf[tup] = support[tup[1]]/support[tup[0]]        
    return conf

def minimal_confidence(conf,min_support):
    return filter(lambda x: conf[x]>min_support,conf)

def graph_data(final_rules,support,convidence):
    supp = []
    conv = []
    for rule in final_rules:
        supp.append(support[rule[1]])
        conv.append(convidence[rule])
    return [supp,conv]

def graph(data,maxX):    
    plt.plot(data[0],data[1],'k.')
    plt.xlim(left=0,right=maxX)
    plt.ylim(bottom=0,top=1.1)
    
def pareto_edge(data):
    edge = dict()         
    data = process_data(data)      
    maxWsparcie = max(data.keys())   
    m = 0    
    for i in range(maxWsparcie,0,-1):        
        if i in data.keys() and data[i]>m:
            edge[i] = data[i]
            m = data[i]
    plt.plot(list(edge.keys()),list(edge.values()),'r.')    
    keys = list(edge.keys())    
    for i, k  in enumerate(keys):         
        if i == 0:
            plt.plot([k,k],[0,edge[k]],':r')
        if i == len(edge.items())-1:
            plt.plot([k,0],[edge[k],edge[k]],':r')
        if i<len(edge.items())-1:
            plt.plot([keys[i+1],k],[edge[keys[i]],edge[keys[i]]],':r')
            plt.plot([keys[i+1],keys[i+1]],[edge[keys[i]],edge[keys[i+1]]],':r')
    return edge
                
def process_data(data):    
    result = dict.fromkeys(data[0],0)    
    for i in range(len(data[0])):        
        if result[data[0][i]]<data[1][i]:
            result[data[0][i]] = data[1][i]
    return result


def apriori(data,minSupport,minConvidance):
    start_time = time.time()
    [sets,supports] = determine_sets(data,minSupport)    
    rules = determine_rules(sets)
    # print('Reguly:\n ',reguly)
    conv = confidence(rules,supports)    
    final_rules = list(minimal_confidence(conv,minConvidance))
    print(' ================ FINAL RULES ================')
    for r in final_rules:
        print_rule(r[0],r[1])
        print('Convidence:',round(conv[r],4))
        print('Support: ',round(supports[r[1]]))
    #wykres
    graph_data_prepared = graph_data(final_rules,supports,conv)
    
    graph(graph_data_prepared,1.1*max(graph_data_prepared[0]))
    edge = pareto_edge(graph_data_prepared)
    print(' ================ RULES ON APRIORI\'S EDGE ================')    
    for k,v in conv.items():
        for b in edge:
            if v == edge[b]:
                print_rule(k[0], k[1])
                print('Convidence: ',edge[b])
                print('Support: ',b)
    
    plt.title('Aprori graph for data set with '+str(data.shape[1])+' attributes\n minimal support: '+str(minSupport)+' \n minimal convidence: '+str(minConvidance))
    plt.show()
    print('================ TIME ================')
    print("Processing time : ",round(time.time()-start_time,2),' s')
    print("Amount of frequent sets: ",len(final_rules))
    


if __name__ == "__main__":
    reuters = sio.loadmat('reuters.mat')
    reuters = reuters['TOPICS'][:1000,:]
    apriori(reuters,15,0.6)
