# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:03:52 2020

@author: Mike Toreno II
"""




import argparse


parser = argparse.ArgumentParser(description = "evaluation")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/preprocessed_data/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument('-n', '--num_components', nargs='+', type = int, default = [5, 10, 15, 20, 30, 40, 50, 60, 75, 100],help="pass the number of compontents to try like this: python script.py --num_compontents 5 10 20 40")
parser.add_argument('--nargs', nargs='+', type=int)
parser.add_argument("-k", "--num_kmclusters", default = 5, help= "number of k for k-means clusters")
parser.add_argument("--reps", default = 100, help= "how many times you want kmcluster to be repeated for each value of num_components")

args = parser.parse_args() #required












##############################################################################
##### Main

def pca_kmc(componentslist = [5, 10, 15, 20, 30, 40, 50, 60, 75, 100],
            input_dir = "../inputs/preprocessed_data/",
             output_dir = "../outputs/hyperparameter/scaPCA_output/",
             num_cluster = 5,
             repetitions = 100
             ):

    
    print("i started ahhaha")

    
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
        print("default k for kmcluster (5) overwritten through command line ({0:d}).".format(num_cluster))
    except:
        print("default k used for kmcluster (5).")
    
    
    

    
    purities = np.zeros((len(componentslist), repetitions))
    errorbars = np.zeros(len(componentslist))
    
    

    
    
    
    for i in range(len(componentslist)):
        
        numcomp = componentslist[i]
        
        intermediate_dir =  output_dir + str(numcomp) + "/"   
        
        if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)
        
        
        commandstring = "python ../2_Baseline_Scripts/sca_PCA.py --num components {numcom:d} --input_dir {inp:s} --output_dir {outp:s} --outputplot_dir {outp:s}".format(numcom = numcomp, inp = input_dir, outp = intermediate_dir)
              
        p1 = subprocess.run(args = commandstring, shell = True, capture_output = True, text = True, check = True)
        
        
        # sca_PCA(num_components = numcomp,
        #         input_path = input_dir,
        #         output_dir = intermediate_dir,
        #         outputplot_dir = intermediate_dir)
        
        
        
        
        # for j in range(repetitions):
        #     current_purity = sca_kmcluster(k = num_cluster,
        #                   dimensions = 0,
        #                   input_path = intermediate_dir,
        #                   output_dir = intermediate_dir,
        #                   outputplot_dir = intermediate_dir + str(j),
        #                   verbosity = 0,
        #                   elbow = False,
        #                   title = "PCA with {0:d} components".format(numcomp),
        #                   reset = False)
        
        #     purities[i, j] = current_purity
            
        #     plt.close('all')
            
            
            
        # errorbars[i] = statistics.stdev(purities[i,:])
    
            
    
    
    
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
    
    
    fake = [3, 5, 7]
    
    
    
    pca_kmc(componentslist = fake,
            input_dir = args.input_dir,
             output_dir = args.output_dir,
             num_cluster = args.num_kmclusters,
             repetitions = args.reps
             )






