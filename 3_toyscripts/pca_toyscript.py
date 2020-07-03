# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 01:25:34 2020

@author: Simon Streib
"""


# %% Load Data
import scipy.io
import numpy as np
import sklearn.decomposition as deco
import matplotlib.pyplot as plt





'''
guide: 
    n_components: the function of this arguments depends heavily on the solver:
        it can e.g. also be used to specify the percentage of variance that is 
        to be kept. Or you can also specify "mle" instead of a number.
        
        The interesting ones are probably:
            - solver = full, n_components 0<n<1. -> then it keeps enough to have
            the percentage of explained variance higher than that. 
            - solver = arpack: the number must be strictly less than n_features
            or n_sapmles. 

    svd_solver: auto, full, arpack, randomized
        - auto is decides based on shape and n_comp.
        - Full solver: exact full SVD with standard LAPACK solver
        - arpack: truncated SVD (to n_components)
        - randomized: an efficient way for 500x500 + size data, and less than
        80% of the components are needed. 
'''


# %% IRIS PCA tutorial

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

x = df.loc[:, ['sepal length', 'sepal width', 'petal length', 'petal width']].values # Separating out the features
y = df.loc[:,['target']].values # Separating out the target


### PCA is affected by scale -> scale the data [mean = 0, sd = 1]
x = StandardScaler().fit_transform(xpre) # Standardizing the features



myPCA = PCA(n_components=4)
PCs = myPCA.fit_transform(x)

# or as a dataframe
principalDf = pd.DataFrame(data = PCs, columns = ['principal component 1', 'principal component 2', 'pc3', "pc4"])
principalDf['target'] = y





fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()





explained_variance = myPCA.explained_variance_ratio_









# %%

df = df.drop(columns = 'target')





myPCA = deco.PCA(n_components=2)
myPCA.fit(df)


print(myPCA.explained_variance_ratio_)






newdata = myPCA.transform(df)



plotdata = newdata
plt.scatter(plotdata[:,0], plotdata[:,1])
plt.show()


# %% Real PCA: (on dense data)
import matplotlib as mp

# Generate Data:
data = np.array([[-1, -1, 1, 3], [-2, -1, 4, 4], [-3, 1, -2, -1], [-4, 1, 1, 2], [2, 1, -2, 3], [3, -1, 2, 4]] )

print(data)


myPCA = deco.PCA(n_components=4)
myPCA.fit(data)

print(myPCA.explained_variance_ratio_)
print(myPCA.singular_values_)

newdata = myPCA.fit_transform(data)



plotdata = data
mp.pyplot.scatter(plotdata[:,0], plotdata[:,1])

plotdata = newdata
mp.pyplot.scatter(plotdata[:,0], plotdata[:,1])

print(newdata)

explained_variance = myPCA.explained_variance_ratio_


### yeah those seem useless
print("--------")
print(myPCA.get_params)


print("--------")
print(myPCA.score)


print("--------")
print(myPCA.score_samples)




























##############################################################################
####### THERE IS ONLY TRASH BEYOND THIS POINT, ENTER AT YOUR OWN RISK ########
##############################################################################



# %% RandomizedPCA DEPRECATED
'''
using it on sparse Data does not "do" PCA, but LCA instead:
    
To clarify: PCA is mathematically defined as centering the data (removing the 
mean value to each feature) and then applying truncated SVD on the centered data.

As centering the data would destroy the sparsity and force a dense representation 
that often does not fit in memory any more, it is common to directly do truncated 
SVD on sparse data (without centering). This resembles PCA but it's not exactly the same
    
some dudes are trying though to do it apparently
https://github.com/scikit-learn/scikit-learn/pull/12841
i don't know how to do this. 
'''



'''
some more infos, when doing PCA: use kernel iteratively
https://stackoverflow.com/questions/13425947/principal-component-analysis-pca-on-huge-sparse-dataset
'''


# %% do SparsePCA
'''
so sklearn.decomposition.PCA does not support sparse input. Instead they propose
to use TruncatedSVD, however this method does not do PCA, but instead LSA. 

I found however sklearn.decomposition.SparsePCA, and I assume this one works with sparse data. 
edit: nope, the sparsity does not refer to the matrix, but instead they try to find
a set of sparse components. 

the whole idea is greatly explained here
https://scikit-learn.org/stable/modules/decomposition.html#sparsepca
'''

# import sklearn.decomposition as deco

# ### SparsePCA
# transformer = deco.SparsePCA(n_components = 5, random_state = 0)
# transformer.fit(coomatrix) ### here we fail :(

# transformed_matrix = transformer.transform(coomatrix)
# transformed_matrix

# %% do normal PCA

# import sklearn.decomposition as deco

# pca = deco.PCA(n_components = 2)

# pca.fit(coomatrix)

# Fails too cuz its too sparse

























