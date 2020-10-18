# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:37:03 2020

@author: Mike Toreno II
"""


import time
import sys
# call script with python script.py var1 var2
import random



z = 20
ele = sys.argv[1]
ele2 = sys.argv[2]



for i in range(z):

    with open("test.tsv", "a") as outfile:
        outfile.write(str(ele) + "\t" + str(ele2))
        outfile.write("\n")
        
    sleepytime = random.uniform(0.00001, 20) 
    time.sleep(1)
    
    

    


































