# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:48:08 2022

@author: stanl
"""
#%% Train Test Split
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
data = pd.read_csv("Wine.csv", header = None)

### train test split label 1 data, label 2 data, label 3 data respectively  
train_1, test_1 = train_test_split(data[data[0] == 1], test_size=18, random_state=10)
train_2, test_2 = train_test_split(data[data[0] == 2], test_size=18, random_state=10)
train_3, test_3 = train_test_split(data[data[0] == 3], test_size=18, random_state=10)

### concatenate label 1, 2 ,3 data and save as csv
train_data = pd.concat([train_1, train_2, train_3], axis = 0)
test_data = pd.concat([test_1, test_2, test_3], axis = 0)
train_data.to_csv("train.csv",index=False ,header=None)
test_data.to_csv("test.csv",index=False, header=None)

#%% Posterior Probability
### Calculate maen and stdev of each label in training data
import math
col1_mean = train_1.mean(axis = 0)
col1_std = train_1.std(axis = 0)
col2_mean = train_2.mean(axis = 0)
col2_std = train_2.std(axis = 0)
col3_mean = train_3.mean(axis = 0)
col3_std = train_3.std(axis = 0)

### Calculate the prior probabilities of 3 labels
prior_1 = len(train_1) / len(train_data)
prior_2 = len(train_2) / len(train_data)
prior_3 = len(train_3) / len(train_data)

### Gaussian Probability Function
def GaussianProbability(mean, std, test):
    temp = math.exp(-(math.pow(test - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * temp

### Calculate the posterior probabilities of each testing instance 
###    by timimg up the prior probability and likelihood function of each column
### Take the greatest among three labels with argmax function as the prediction 
correct_prediction = 0
for i in range(len(test_data)):
    label1_prob = prior_1
    label2_prob = prior_2
    label3_prob = prior_3
    for j in range(1, 14):    ### column 0 is the label, so shall not be included
        label1_prob *= GaussianProbability(col1_mean[j], col1_std[j], test_data.iloc[i][j].item())
        label2_prob *= GaussianProbability(col2_mean[j], col2_std[j], test_data.iloc[i][j].item())
        label3_prob *= GaussianProbability(col3_mean[j], col3_std[j], test_data.iloc[i][j].item())
    labels_prob = [label1_prob, label2_prob, label3_prob]
    if test_data.iloc[i][0].item() == (np.argmax(labels_prob) + 1):
        correct_prediction += 1

### Evaluate the accuracy
print("----Evaluation of Posterior Probability Algorithm----")
print("Accuracy: " + str(100.0 * correct_prediction / len(test_data)))

#%% Visualized Result
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

### Fit pca with the training set and transform on the testing set
test_data_nolabel = test_data.iloc[:, 1:]
test_data_label = test_data.iloc[:, 0]
train_data_nolabel = train_data.iloc[:, 1:]
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
_ = pipe.fit(train_data_nolabel)
X_pca = pipe.transform(test_data_nolabel)

### Plot the first two components on 2-d scatter plot
plot = plt.scatter(X_pca[:,0], X_pca[:,1], c=test_data_label)
plt.legend(handles=plot.legend_elements()[0], labels=list((1,2,3)))
plt.show()

#%% Effect of Prior Probability
### Add a variable to evaluate the accuracy without prior probability
correct_prediction = 0
correct_prediction_without_prior = 0

### Here, I didn't include every feature, because the effect of prior probability can't be seen with too many features.
### Tried calculating probability with and without prior probability
import random
l = list(i for i in range(1, 14))
random.shuffle(l)
for i in range(len(test_data)):
    label1_prob = prior_1
    label2_prob = prior_2
    label3_prob = prior_3
    label1_prob_without_prior = 1
    label2_prob_without_prior = 1
    label3_prob_without_prior = 1
    for j in l[:2]:
        label1_prob *= GaussianProbability(col1_mean[j], col1_std[j], test_data.iloc[i][j].item())
        label2_prob *= GaussianProbability(col2_mean[j], col2_std[j], test_data.iloc[i][j].item())
        label3_prob *= GaussianProbability(col3_mean[j], col3_std[j], test_data.iloc[i][j].item())
        label1_prob_without_prior *= GaussianProbability(col1_mean[j], col1_std[j], test_data.iloc[i][j].item())
        label2_prob_without_prior *= GaussianProbability(col2_mean[j], col2_std[j], test_data.iloc[i][j].item())
        label3_prob_without_prior *= GaussianProbability(col3_mean[j], col3_std[j], test_data.iloc[i][j].item())
    labels_prob = [label1_prob, label2_prob, label3_prob]
    labels_prob_without_prior = [label1_prob_without_prior, label2_prob_without_prior, label3_prob_without_prior]
    if test_data.iloc[i][0].item() == (np.argmax(labels_prob) + 1):
        correct_prediction += 1
    if test_data.iloc[i][0].item() == (np.argmax(labels_prob_without_prior) + 1):
        correct_prediction_without_prior += 1
        
### Both evaluations
print("----Effect of Prior Probability with Random Features----")
print("Accuracy with prior: " + str(100.0 * correct_prediction / len(test_data)))
print("Accuracy without prior: " + str(100.0 * correct_prediction_without_prior / len(test_data)))

#%% Effect of Prior Probability with PCA
### Fit PCA on training set and transform on testing set
### Calculate the new mean and std
train_data_nolabel = train_data.iloc[:, 1:]
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=1))])
_ = pipe.fit_transform(train_data_nolabel)
test_pca = pipe.transform(test_data_nolabel)
col1_mean_pca = _[train_data[0] == 1].mean(axis = 0)
col1_std_pca = _[train_data[0] == 1].std(axis = 0)
col2_mean_pca = _[train_data[0] == 2].mean(axis = 0)
col2_std_pca = _[train_data[0] == 2].std(axis = 0)
col3_mean_pca = _[train_data[0] == 3].mean(axis = 0)
col3_std_pca = _[train_data[0] == 3].std(axis = 0)

### Add a variable to evaluate the accuracy without prior probability
correct_prediction_pca = 0
correct_prediction_without_prior_pca = 0

### Here, I only include the significant component of PCA-transformed data.
### Tried calculating probability with and without prior probability
for i in range(len(test_pca)):
    label1_prob = prior_1
    label2_prob = prior_2
    label3_prob = prior_3
    label1_prob_without_prior = 1
    label2_prob_without_prior = 1
    label3_prob_without_prior = 1
    for j in range(1):
        label1_prob *= GaussianProbability(col1_mean_pca[j], col1_std_pca[j], test_pca[i][j].item())
        label2_prob *= GaussianProbability(col2_mean_pca[j], col2_std_pca[j], test_pca[i][j].item())
        label3_prob *= GaussianProbability(col3_mean_pca[j], col3_std_pca[j], test_pca[i][j].item())
        label1_prob_without_prior *= GaussianProbability(col1_mean_pca[j], col1_std_pca[j], test_pca[i][j].item())
        label2_prob_without_prior *= GaussianProbability(col2_mean_pca[j], col2_std_pca[j], test_pca[i][j].item())
        label3_prob_without_prior *= GaussianProbability(col3_mean_pca[j], col3_std_pca[j], test_pca[i][j].item())
    labels_prob = [label1_prob, label2_prob, label3_prob]
    labels_prob_without_prior = [label1_prob_without_prior, label2_prob_without_prior, label3_prob_without_prior]
    if test_data.iloc[i][0].item() == (np.argmax(labels_prob) + 1):
        correct_prediction_pca += 1
    if test_data.iloc[i][0].item() == (np.argmax(labels_prob_without_prior) + 1):
        correct_prediction_without_prior_pca += 1
        
### Both evaluations
print("----Effect of Prior Probability with PCA----")
print("Accuracy with prior: " + str(100.0 * correct_prediction_pca / len(test_data)))
print("Accuracy without prior: " + str(100.0 * correct_prediction_without_prior_pca / len(test_data)))