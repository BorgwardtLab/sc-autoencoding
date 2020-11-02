# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:03:52 2020

@author: Mike Toreno II
"""

import argparse


parser = argparse.ArgumentParser(description = "evaluation")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/data/preprocessed_data/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../outputs/optimization/num_PCA/")
parser.add_argument('-n', '--num_components', nargs='+', type = int, default = [5, 10, 15, 20, 30, 40, 50, 60, 75, 100],help="pass the number of components to try like this: python script.py --num_components 5 10 20 40")
parser.add_argument("-k", "--num_kmclusters", default = 5, help= "number of k for k-means clusters", type = int)
parser.add_argument("--reps", default = 100, help= "how many times you want kmcluster to be repeated for each value of num_components", type = int)

args = parser.parse_args() #required







##############################################################################
##### Main

def pca_kmc(componentslist = [5, 10, 15, 20, 30, 40, 50, 60, 75, 100],
            input_dir = "../inputs/data/preprocessed_data/",
             output_dir = "../outputs/optimization/num_PCA/",
             num_cluster = 5,
             repetitions = 100
             ):

    
    print("pca-kmcluster_evaluation started")

    
    # %%
    import sys
    import os
    
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        pass
       
    

    import matplotlib.pyplot as plt
    import statistics
    import numpy as np
    
    import subprocess
    
    
         
    # %% 

    try:
        num_clusters = sys.argv[1]
        print("using k = {0:d}.".format(num_cluster))
    except:
        print("default k used for kmcluster (5).")
    

    
    purities = np.zeros((len(componentslist), repetitions))
    errorbars = np.zeros(len(componentslist))
    
    
    
    
    for i in range(len(componentslist)):
        
        numcomp = componentslist[i]
        
        intermediate_dir =  output_dir + str(numcomp) + "/"   
        
        
        if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)
        
        
        commandstring = "python ../2_Baseline_Scripts/sca_PCA.py --num_components {numcom:d} --input_dir {inp:s} --output_dir {outp:s} --outputplot_dir {outp:s}".format(numcom = numcomp, inp = input_dir, outp = intermediate_dir)
        p1 = subprocess.run(args = commandstring, shell = True, capture_output = True, text = True, check = False)

        print(p1.stdout)
        print(p1.stderr)
        
        # sca_PCA(num_components = numcomp,
        #         input_path = input_dir,
        #         output_dir = intermediate_dir,
        #         outputplot_dir = intermediate_dir)
        
        
        
        
        for j in range(repetitions):
            
            command = "python ../4_Evaluation/sca_kmcluster.py --title {title:s} --k {k:d} --verbosity 0 --dimensions 0 --input_dir {inp:s} --output_dir {outp:s} --outputplot_dir {outplot:s}".format(k = num_cluster, inp = intermediate_dir, outp = intermediate_dir, outplot = intermediate_dir + str(j), title = "PCA_with_{0:d}_components".format(numcomp))
            print(command)
            
            p2 = subprocess.run(args = command, shell = True, capture_output = True, text = True, check = False)


            print(p2.stdout)
            print(p2.stderr)
            
            
            idx_of_purity = p2.stdout.find("global purity is: ")
            
            floatstring = p2.stdout[idx_of_purity+18:idx_of_purity+24]
            current_purity = float(floatstring)


        
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
    



# %%


if __name__ == "__main__":
    
    
   
    pca_kmc(componentslist = args.num_components,
            input_dir = args.input_dir,
             output_dir = args.output_dir,
             num_cluster = args.num_kmclusters,
             repetitions = args.reps
             )






