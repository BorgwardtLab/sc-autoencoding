# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:37:03 2020

@author: Mike Toreno II
"""


import time


z = 20
number = 333333333333333333333333333333333333333333333333333333333333333333333


for i in range(z):

    
    
    with open("test.tsv", "a") as outfile:
        outfile.write(str(number))
        outfile.write("\n")
    time.sleep(6.2)
    
    
    
print("I ended " + str(number))    
    
    
    









































