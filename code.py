# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:52:22 2015

@author: paresh
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/paresh/anaconda/lib/python2.7/site-packages/lib/python2.7/site-packages')
from PyML import *
from sklearn import cross_validation
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from copy import deepcopy
from sklearn import preprocessing

"""Read data"""
X_data, y_data = load_svmlight_file("data/scop_motif_preprocessed.data")

"""Defining classifiers and calculating accuracy"""

poly_accuracy = []
gaussian_accuracy = []
gamma_arr = []
poly_deg = []
soft_margin = []
gamma_temp=0.0001
k=1
for i in range (1,9):
    C_temp=0.0001
    for j in range (1,9):
        #defining polynomial kernel classifier
        classifier_poly = svm.SVC(kernel='poly', degree = k,coef0 = 1, C=C_temp )       
        #defining gaussian kernel classifier
        classifier_gaussian = svm.SVC(kernel='rbf', gamma = gamma_temp, C=C_temp )        
        # performing cross validation:
        poly_accuracy.append(np.mean(cross_validation.cross_val_score(classifier_poly, X_data, y_data, cv=5, scoring='roc_auc')))
        gaussian_accuracy.append(np.mean(cross_validation.cross_val_score(classifier_gaussian, X_data, y_data, cv=5, scoring='roc_auc')))
        gamma_arr.append(C_temp)
        poly_deg.append(k)
        C_temp=C_temp*10
    gamma_temp=gamma_temp*10
    if (i<5):
        k=k+1
    else:
        k=k+5

"""Graph plotting of roc-Accuracy"""

#Plotting 3d data for Polynomial
fig = plt.figure()
ax=fig.gca(projection='3d')
X = np.reshape(gamma_arr,(-1,8))
X=np.log10(X)
Y = np.reshape(poly_deg,(-1,8))
Z = np.reshape(poly_accuracy,(-1,8))
xticks = [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]
ax.set_xticks(np.log10(xticks))
ax.set_xticklabels(xticks)
yticks = [1,2,3,4,5,10,15,20]
ax.set_yticks(yticks)
ax.set_title("Polynomial kernel: roc_Accuracy vs kernel parameter vs soft margin")
ax.set_xlabel("Soft margin parameter C")
ax.set_ylabel("Degree")
ax.set_zlabel("roc_Accuracy")
ax.plot_surface(X, Y, Z, cmap='Accent',cstride=1, rstride=1,linewidth=0.5,shade=True,alpha=0.7)
plt.show()

#Plotting 3d data for gaussian
fig = plt.figure()
ax=fig.gca(projection='3d')
X = np.reshape(gamma_arr,(-1,8))
X=np.log10(X)
Y_gauss = np.reshape(gamma_arr,(-1,8))
Y_gauss=np.log10(Y_gauss)
Y_gauss=np.transpose(Y_gauss)
Z_gauss = np.reshape(gaussian_accuracy,(-1,8))
xticks = [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]
ax.set_xticks(np.log10(xticks))
ax.set_xticklabels(xticks)
yticks = [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]
ax.set_yticks(np.log10(yticks))
ax.set_yticklabels(yticks)
ax.set_title("Gaussian kernel: roc_Accuracy vs kernel parameter vs soft margin")
ax.set_xlabel("Soft margin parameter C")
ax.set_ylabel("Gamma")
ax.set_zlabel("roc_Accuracy")
ax.plot_surface(X, Y_gauss, Z_gauss, cmap='Accent',cstride=1, rstride=1,linewidth=0.5,shade=True,alpha=0.7)
plt.show()

#Plotting 2d cross sections for polynomial and gaussian accuracy vs kernel parameter
fig=plt.figure()
for i in range(len(X[0])):
    twod_val_Y=[]
    twod_val_Z=[]
    twod_val_Y_gauss=[]
    twod_val_Z_gauss=[]
    for j in range(len(X)):
        twod_val_Y.append(Y[j][i])
        twod_val_Z.append(Z[j][i])
        twod_val_Y_gauss.append(Y_gauss[j][i])
        twod_val_Z_gauss.append(Z_gauss[j][i])
    plt.subplot(2,2,1)
    plt.xlabel("degree of polynomial kernel")
    plt.ylabel("roc_Accuracy")
    plt.title("roc_Accuracy vs kernel parameter for Polynomial kernel")
    plt.plot(twod_val_Y,twod_val_Z)
    plt.subplot(2,2,2)
    plt.xlabel("gamma of gaussian kernel")
    plt.ylabel("roc_Accuracy")
    plt.title("roc_Accuracy vs kernel parameter for Gaussian kernel")
    plt.plot(twod_val_Y_gauss,twod_val_Z_gauss)

#Plotting 2d cross sections for polynomial and gaussian accuracy vs soft margin
for i in range(len(Y)):
    twod_val_X=[]
    twod_val_Z=[]
    twod_val_Z_gauss=[]
    for j in range(len(X[0])):
        twod_val_X.append(X[i][j])
        twod_val_Z.append(Z[i][j])
        twod_val_Z_gauss.append(Z_gauss[i][j])
    plt.subplot(2,2,3)
    plt.xlabel("soft margin")
    plt.ylabel("roc_Accuracy")
    plt.title("roc_Accuracy vs soft margin for Polynomial kernel")
    plt.plot(twod_val_X,twod_val_Z)
    plt.subplot(2,2,4)
    plt.xlabel("soft margin")
    plt.ylabel("roc_Accuracy")
    plt.title("roc_Accuracy vs soft margin for Gaussian kernel")
    plt.plot(twod_val_X,twod_val_Z_gauss)
plt.show()


"""Accuracy of SVM on raw vs normalized"""

#data normalization
X_data_norm = deepcopy(preprocessing.normalize(X_data))

#Grid search on raw data
Cs_arr=[0.0001,0.001,0.01,0.1,1,10,100,1000]
gammas_arr=[0.0001,0.001,0.01,0.1,1,10,100,1000]
raw_score=[]
norm_score=[]
for i in range(1,9):
    param_grid=[{'C':Cs_arr,'gamma':gammas_arr}]
    classifier = GridSearchCV(estimator=svm.SVC(kernel='rbf'), param_grid=param_grid)
    raw_score.append(np.mean(cross_validation.cross_val_score(classifier, X_data, y_data, cv=5,scoring='roc_auc')))
    norm_score.append(np.mean(cross_validation.cross_val_score(classifier, X_data_norm, y_data, cv=5, scoring='roc_auc')))   
    del Cs_arr[0]
    del gammas_arr[0]
plt.figure()
plt.plot(raw_score)
plt.plot(norm_score)
plt.ylabel("roc_Accuracy")
plt.title("Comparison of roc_Accuracy of raw and normalized data")
plt.show()

"""Kernel Plot"""
data = SparseDataSet('data/scop_motif.data')
ker.showKernel(data)
data.normalize()
ker.showKernel(data)