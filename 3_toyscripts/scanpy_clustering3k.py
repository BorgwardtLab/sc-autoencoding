# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:59:05 2020

@author: Mike Toreno II
"""

import scanpy as sc

import pickle
file = open("D:/Dropbox/Internship/gitrepo/3_toyscripts/objectsavefile.obj", "rb")
obi = pickle.load(file)


AnnData = 1




# %% PCA

# do PCA
sc.tl.pca(AnnData, svd_solver='arpack')


# do Scatterplot
sc.pl.pca(AnnData, color='CST3')
plt.savefig(outputplot_dir + "PCA_plot.png")
   

# Scree PLot
sc.pl.pca_variance_ratio(AnnData, log=True)
plt.savefig(outputplot_dir + "SCA_plot.png")
   

# Save results
AnnData.write(results_file)







# %% Neighbourhood graph

# compute neighborhood graph
sc.pp.neighbors(AnnData, n_neighbors=10, n_pcs=40)


# embedd the neighborhood graph


# this section is to remedy if umap global connection is bad
# # -> embed it using UMAP
# tl.paga(AnnData)
# pl.page(AnnData) # THIS LINE SHOULD BE REMOVED ITS JUST TO SHOW THE COARSE GRAINED PICTURE
# pl.paga(AnnData, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
# tl.umap(AnnData, init_pos='paga')


# compute UMAP
sc.tl.umap(AnnData)

# plotting the .RAW (default)
sc.pl.umap(AnnData, color=['CST3', 'NKG7', 'PPBP'])

# plotting the corrected & scaled
sc.pl.umap(AnnData, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)



### Clustering the neighborhood graph
sc.tl.leiden(AnnData)


sc.pl.umap(AnnData, color=['leiden', 'CST3', 'NKG7'])


# save the result
AnnData.write(results_file)







