#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
import numpy as np
import os
import sklearn
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


# sentiment analysis
analyser = SentimentIntensityAnalyzer()
# chosing the limit values is a hyper parameter which will be run in a grid search
def sentiment_analyzer_scores(sentence,a,b):
    score = analyser.polarity_scores(sentence)
    if score['neg'] > a+0.001:
        score_m=0
    elif score['pos'] > a+0.001:
        score_m=4
#     elif score['neu'] > b or (score['pos']<a and score['neg']<a): 
#         score_m=2
    else:
        score_m=0
    return score_m


# In[3]:


# generate new test data - originally used for validation
with open('C:/Users/jonat/Documents/School/NLP/FinalProject/Training_Data/training.1600000.processed.noemoticon.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    train_labels=[]
    train_data=[]
    for row in csv_reader:
        train_labels.append(row[0])
        train_data.append(row[5])
        line_count += 1
#     print(f' We have {line_count} samples')
# define a subset of train data for testing
# note that as this algorithm was pretrained, there was no training at all
# variable names kept the same for consistency
augmented_testdata=train_data[:500000]
augmented_testlabels=train_labels[:500000]


# In[3]:


def predict_results(test_data,test_labels,a,b):
    data_prediction=[]
    for i in range(len(test_data)):
        temp=0
        temp=sentiment_analyzer_scores(test_data[i],a,b)
        data_prediction.append(str(temp))
    accuracy=accuracy_score(test_labels,data_prediction)
    return accuracy


# In[6]:


# acc1=predict_results(test_data,test_labels)
# print(acc1)
a=0.02
b=0.25
# acc_augmented=predict_results(augmented_testdata,augmented_testlabels,a,b)
acc_augmented=predict_results(train_data,train_labels,a,b)
# output validation accuracy here
print(acc_augmented) 


# In[ ]:


# grid search
# a parameter= pos/neg boundary
# b parameter= neutral boundary
def gridsearch():
    d=(70,35)
    acc_augmented = np.zeros(d)
    for a in range(0,70,1):
        for b in range(0,35,1):      
            acc_augmented[a][b]=predict_results(augmented_testdata,augmented_testlabels,a,b)
            print(a/100, b/100, acc_augmented[a][b])
            
    return acc_augmented
t=gridsearch()


# In[8]:


# We apply this algorithm to classify our unlabelled tweets

file_path_X = "C:/Users/jonat/Documents/School/NLP/FinalProject/Train_plus_Test/direct_X_FULL.pkl"
X = pickle.load( open(file_path_X, "rb" ) )    
file_path_Y = "C:/Users/jonat/Documents/School/NLP/FinalProject/Train_plus_Test/direct_Y_FULL.pkl"
Y = pickle.load( open(file_path_Y, "rb" ) ) 


# In[9]:


tweets=[]
for i in range(len(X)):
    tweets.append(X[i][0])


# In[10]:


def predict_results_unlabeled(test_data,a,b):
    data_prediction=[]
    for i in range(len(test_data)):
        temp=0
        temp=sentiment_analyzer_scores(test_data[i],a,b)
        data_prediction.append(str(temp))
    return data_prediction


# In[11]:


# for prediction
a=0.02
b=0.25
predicted_labels_unlabeled=predict_results_unlabeled(tweets,a,b)
# sentiment vector will be divided as necessary by the baseline classifiers


# In[14]:


pickle.dump(predicted_labels_unlabeled, open('Vader_labelprediction.pickle', 'wb'))

