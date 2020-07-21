# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:03:52 2020

@author: Mike Toreno II
"""

# %%
import sys
import os

try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
   

sys.path.insert(1, "../2_Baseline_Scripts")
from sca_PCA import sca_PCA
sys.path.insert(1, "../1_Main_Scripts")
from sca_kmcluster import sca_kmcluster


import matplotlib.pyplot as plt
import statistics
import numpy as np




     
# %% 
num_cluster = 5
repetitions = 100


try:
    num_clusters = sys.argv[1]
    print("default k for kmcluster (5) overwritten through command line ({0:d}).".format(num_cluster))
except:
    print("default k used for kmcluster (5).")

             



componentslist = [5, 10, 15, 20, 30, 40, 50, 60, 75, 100]

purities = np.zeros((len(componentslist), repetitions))
errorbars = np.zeros(len(componentslist))


input_dir = "../inputs/preprocessed_data/"
output_dir = "../outputs/hyperparameter/scaPCA_output/"



for i in range(len(componentslist)):
    
    numcomp = componentslist[i]
    
    intermediate_dir =  output_dir + str(numcomp) + "/"   
    
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    
    
    sca_PCA(num_components = numcomp,
            input_path = input_dir,
            output_dir = intermediate_dir,
            outputplot_dir = intermediate_dir)
    
    
    for j in range(repetitions):
        current_purity = sca_kmcluster(k = num_cluster,
                      dimensions = 0,
                      input_path = intermediate_dir,
                      output_dir = intermediate_dir,
                      outputplot_dir = intermediate_dir + str(j),
                      verbosity = 0,
                      elbow = False,
                      title = "PCA with {0:d} components".format(numcomp),
                      reset = False)
    
        purities[i, j] = current_purity
        
        plt.close('all')
        
        
        
    errorbars[i] = statistics.stdev(purities[i,:])

        



# %%


averages = np.average(purities, axis = 1)

plt.figure()
for i, (x,y) in enumerate(zip(componentslist, averages)):
    
    label = "{:.5f}".format(y)
    
    plt.annotate(label, # this is the text
             (x,y), # this is the point to label
             textcoords="offset points", # how to position the text (if xytesxt is points or pixels)
             xytext=(30,5), # distance from text to points (x,y)
             ha='center') # horizontal alignment can be left, right or center
    
    plt.errorbar(x,y,yerr= errorbars[i])

plt.plot(componentslist, averages, linestyle = "-", marker = "o")
plt.title("global purities for different number of PCs")
plt.xlabel("number of principal components calculated")
plt.ylabel("global purity evaluated by kmcluster")
plt.show()
plt.savefig(output_dir + "puritygraph.png")



np.savetxt(output_dir + "purities.csv", purities, delimiter = "\t")








