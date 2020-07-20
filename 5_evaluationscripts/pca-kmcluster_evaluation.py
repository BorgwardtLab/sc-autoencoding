# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:03:52 2020

@author: Mike Toreno II
"""

# %%

import sys
sys.path.insert(1, "../2_Baseline_Scripts")
from sca_PCA import sca_PCA
sys.path.insert(1, "../1_Main_Scripts")
from sca_kmcluster import sca_kmcluster


import matplotlib.pyplot as plt
import os
import numpy as np


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
   

     
# %% 

num_cluster = 5
 
try:
    num_clusters = sys.argv[1]
    print("default k for kmcluster (5) overwritten through command line ({0:d}).".format(num_cluster))
except:
    print("default k used for kmcluster (5).")

             


componentslist = [5, 10, 15, 20, 30, 40, 50, 60, 75, 100]
puritylist = np.zeros(len(componentslist))

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
    
    current_purity = sca_kmcluster(k = num_cluster,
                  dimensions = 0,
                  input_path = intermediate_dir,
                  output_dir = intermediate_dir,
                  outputplot_dir = intermediate_dir,
                  verbosity = 0,
                  elbow = False,
                  title = "PCA with {0:d} components".format(numcomp),
                  reset = True)
    
    puritylist[i] = current_purity
    
    plt.close('all')
        
        



# %%


plt.figure()
plt.plot()
plt.plot(componentslist, puritylist, "r-")
plt.title("global purities for different number of PCs")
plt.xlabel("number of principal components calculated")
plt.ylabel("global purity evaluated by kmcluster")
plt.show()
plt.savefig(output_dir + "puritygraph.png")













