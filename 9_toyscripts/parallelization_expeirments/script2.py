# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:37:03 2020

@author: Mike Toreno II
"""


import time


z = 20
number = 222222222222222222222222222222222222222222222222222222222222222222222


for i in range(z):

    
    
    with open("test.tsv", "a") as outfile:
        outfile.write(str(number))
        outfile.write("\n")
    time.sleep(0.87)
    
    
    
print("I ended " + str(number))    
    
    
    









































