# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:37:03 2020

@author: Mike Toreno II
"""


import time
import sys
# call script with python script.py var1 var2



z = 20
number = sys.argv[1]


for i in range(z):

    
    
    with open("test.tsv", "a") as outfile:
        outfile.write(str(number))
        outfile.write("\n")
    time.sleep(1)
    
    
    
print("I ended " + str(number))    
    
    
    


































