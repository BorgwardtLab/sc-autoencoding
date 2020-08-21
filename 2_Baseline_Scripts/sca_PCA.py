# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 01:25:34 2020

@author: Simon Streib
"""


# %% Load Data



import argparse


parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-n","--num_components", help="the number of PCAs to calculate", type = int, default = 30)
parser.add_argument("-s", "--nosave", help="passing this flag prevents the program from saving the reduced coordinates to prevent storage issues. (plots and other output still gets saved)", action="store_true")
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/data/preprocessed_data/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/baselines/baseline_data/scaPCA_output/")
args = parser.parse_args() #required





def sca_PCA(input_path = "../inputs/data/preprocessed_data/",
              output_dir = "../inputs/baselines/baseline_data/scaPCA_output/",
              outputplot_dir = "../outputs/baselines/baseline_data/scaPCA_output/",
              nosave = False,
              num_components = 30
              ):
 
    
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    import matplotlib.cm as cm # colourpalette
    import sys
    
    
    
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        pass
             
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_PCA.py with num_components = {numcom:d}".format(numcom = num_components))    
    component_name = "PC"    
    
    
    
    
    
    # %% Read Input data
    print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")
    
    data = np.loadtxt(open(input_path + "matrix.tsv"), delimiter="\t")
    
    genes = pd.read_csv(input_path + "genes.tsv", delimiter = "\t", header = None)
    
    barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)
    labels = barcodes.iloc[:,1]
    
    
    test_index = np.loadtxt(fname = input_path + "test_index.tsv", dtype = bool)
    train_index = np.logical_not(test_index)
    
    
    
    
    
    ##############################################################################
    # from here on out: 
        # data = traindata
        # testdata = testdata
    complete_data = data
    testdata = data[test_index]
    data = data[train_index]
    ##############################################################################
    
    
    
    ### Get Labels 
    # print(datetime.now().strftime("%H:%M:%S>"), "reading labels...")
    # file = open(input_path + "celltype_labels.tsv", "r")
    # labels = file.read().split("\n")
    # file.close()
    # labels.remove("") #last, empty line is also removed
    ### load barcodes
    # file = open(input_path + "barcodes.tsv", "r")
    # barcodes = file.read().split("\n")
    # file.close()
    # barcodes.remove("") 
    
    
    
    # %%  Cut back data for handlability lmao
    # print(datetime.now().strftime("%H:%M:%S>"), "reading input matrix...")
    # ### Get Matrix
    # mtx_file = input_path + "matrix.mtx"
    # coomatrix = scipy.io.mmread(mtx_file)
    # coomatrix_t = np.transpose(coomatrix)
    
    # print(datetime.now().strftime("%H:%M:%S>"), "deleting random data pieces...")
    # genes_uplimit = 30000
    # genes_downlimit = 25000
    # cells_uplimit = 15000
    # cells_downlimit = 10000
    # labels = labels[cells_downlimit:cells_uplimit]
    # genes = genes[genes_downlimit:genes_uplimit]
    # csrmatrix = coomatrix_t.tocsr()
    # coomatrix_t = csrmatrix[cells_downlimit:cells_uplimit, genes_downlimit:genes_uplimit]
    
    
    # Convert to dense
    # print(datetime.now().strftime("%H:%M:%S>"), "converting sparse matrix to dense...")
    #data = coomatrix_t.toarray()
    
    
    
    
    
    
    # %% do PCA
    
    print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
    
    myscaler = StandardScaler()
    data =  myscaler.fit_transform(data)
    
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "calculating principal components...")
    myPCA = PCA(n_components=num_components)
    PCs = myPCA.fit_transform(data)
    
    
    explained_variance = myPCA.explained_variance_ratio_
    
    
    
    
    
    #################### do testdata
    test_PCs = myscaler.transform(testdata)
    test_PCs = myPCA.transform(test_PCs)
    
    
    
    
    
    
    # %% i know this is messy, but this section here tries to combine the test and traindata into a single output matrix again.
    
    
    # I keep it as a np array here. Other scripts maybe get a dataframe
    #outdata = pd.DataFrame(data = np.zeros(shape = (len(complete_data), num_components)))
    
    
    
    outdata = np.zeros(shape = (len(complete_data), num_components))
    
    outdata[train_index] = PCs
    outdata[test_index] = test_PCs
    
    
    
    
    #%% Outputs
    
    
    
    if not os.path.exists(outputplot_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Plot Directory...")
        os.makedirs(outputplot_dir)
        
    
    
    ### Create Plot
    print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
    targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*
    
    # construct dataframe for 2d plot
    df = pd.DataFrame(data = PCs[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
    df['celltype'] = labels
    
    
    
    
    # %% plots 
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(component_name + '_1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
    ax.set_ylabel(component_name + '_2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
    ax.set_title('Most Powerful PCAs', fontsize = 20)
    colors = cm.rainbow(np.linspace(0, 1, len(targets)))
    for target, color in zip(targets,colors):
        indicesToKeep = df['celltype'] == target
        ax.scatter(df.loc[indicesToKeep, component_name + '_1']
                    , df.loc[indicesToKeep, component_name + '_2']
                    , c = color.reshape(1,-1)
                    , s = 5)
    ax.legend(targets)
    ax.grid()
    plt.savefig(outputplot_dir + "PCA_plot_trainingdata.png")
    
    
    
    
    
    ### Save Variances
    print(datetime.now().strftime("%H:%M:%S>"), "saving explained variances...")
    explained_sum = np.cumsum(explained_variance)
    
    file = open(outputplot_dir + 'explained_variances.log', 'w')
    for i in range(len(explained_variance)):
        text = (str(i + 1) + "\t" + str(explained_variance[i]) + "\t" + str(explained_sum[i]) + "\n")
        file.write(text)
    file.close()
        
        
        
        
        
    ### Scree Plots
    perc_var = (explained_variance * 100)
    perc_var = perc_var[0:num_components]
    
    labelz = [str(x) for x in range(1, len(perc_var)+1)]
    
    
    plt.figure(figsize=[16,8])
    plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
    plt.ylabel('Percentage of explained variance')
    plt.xlabel('Principal component')
    plt.title('Scree plot')
    plt.show()    
    plt.savefig(outputplot_dir + "PCA_scree_plot_all.png")
        
        
        
        
    if num_components > 50:
        how_many = 50;
        perc_var = (explained_variance * 100)
        perc_var = perc_var[0:how_many]
    
        labelz = [str(x) for x in range(1, len(perc_var)+1)]
        
        plt.figure(figsize=[16,8])
        plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
        plt.ylabel('Percentage of explained variance')
        plt.xlabel('Principal component')
        plt.title('Scree plot')
        plt.show()    
        plt.savefig(outputplot_dir + "PCA_scree_plot_top50.png")    
        
        
           
        
    # Loading scores for PC1
    
    how_many = 10
    
    loading_scores = pd.Series(myPCA.components_[0], index = genes.iloc[:,1])
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_genes = sorted_loading_scores[0:how_many].index.values
        
    
    file = open(outputplot_dir + 'most_important_genes.log', 'w')
    for i in range(how_many):
        text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
        file.write(text)
    file.close()
    
    
    
    
    
    
    
    
    
    
    # %% Saving the data
    
    if nosave == False:
    
        if not os.path.exists(output_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
            os.makedirs(output_dir)    
    
        print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
      
        
      
        np.savetxt(output_dir + "matrix.tsv", outdata, delimiter = "\t")
            
        genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
        
        barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)
        
        np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")
        
        # with open(output_dir + "barcodes.tsv", "w") as outfile:
        #     outfile.write("\n".join(barcodes))
        
        # with open(output_dir + "celltype_labels.tsv", "w") as outfile:
        #     outfile.write("\n".join(labels))
    
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "sca_PCA.py terminated successfully\n")




#------------------------------------------------------------------------------
# run
#------------------------------------------------------------------------------



if __name__ == "__main__":
    sca_PCA(input_path = args.input_dir, output_dir= args.output_dir, 
              outputplot_dir= args.outputplot_dir, nosave = args.nosave, 
              num_components= args.num_components)













