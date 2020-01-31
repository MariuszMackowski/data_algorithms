# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:16:35 2019

@author: Mario
"""

import random

def losowa():
    while True:
        x = round(random.uniform(100,999))            
        if '7' in str(x) and '8' not in str(x):            
            return x

def suma(x,y,z):
    return x+y == z
         

while True:
    x = losowa()
    y = losowa()
    z = losowa()
    if suma(x,y,z):
        print(x,' + ',y,' = ',z)
        break