#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import os
import sklearn
from sklearn.utils import shuffle
import pickle
import random
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


#  load train and test data
with open('C:/Users/jonat/Documents/School/NLP/FinalProject/Training_Data/training.1600000.processed.noemoticon.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    train_labels_=[]
    train_data_=[]
    for row in csv_reader:
        train_labels_.append(row[0])
        train_data_.append(row[5])
        line_count += 1


indexes = list(range(len(train_data_)))
random.shuffle(indexes)
train_data = [train_data_[indexes[i]] for i in range(len(indexes))]
train_labels = [train_labels_[indexes[i]] for i in range(len(indexes))]
augmented_testdata=train_data[15*100000:16*100000]
augmented_testlabels=train_labels[15*100000:16*100000]


# In[3]:


# vectorize text, with N grams etc
# tunned exhaustively
Threshold =1
vectorizer = CountVectorizer(encoding='ascii', min_df=Threshold, ngram_range=(1,2), decode_error='ignore', analyzer='word')
features=vectorizer.fit_transform(train_data[:100000])
vocab = vectorizer.vocabulary_
test_data_vec=vectorizer.transform(augmented_testdata)


# In[4]:


# pickle dump to store original training data
#loop
for i in range(16):
    vectorizer = CountVectorizer(encoding='ascii', min_df=Threshold, ngram_range=(1,2), decode_error='ignore', analyzer='word', vocabulary = vocab)
    features=vectorizer.transform(train_data[(i)*100000:(i+1)*100000])
    pickle.dump(features, open('train'+str(i)+'.pickle', 'wb'))
    print(i,'Complete')


# In[5]:


class NaiveBayes():
    def __init__(self):
        self.labels_predict=[]
        self.accuracy=0
        self.Talg=0
    def train(self): 
        self.Talg = MultinomialNB()
        for i in range(14):
            data_train = pickle.load(open('train'+str(i)+'.pickle',"rb"))
            labels_train=train_labels[(i)*100000:(i+1)*100000]
            self.Talg = self.Talg.partial_fit(data_train,labels_train,classes=np.unique(labels_train))
    def test(self):
        self.labels_predict=self.Talg.predict(test_data_vec)
        self.accuracy=accuracy_score(augmented_testlabels,self.labels_predict)
        return self.accuracy, self.labels_predict
    def unlabelled_prediction(self,tweets):
        self.labels_predict=self.Talg.predict(tweets)
        return self.labels_predict


# In[6]:


NB1=NaiveBayes()
NB1.train()
a,b=NB1.test() 
# validation/test accuracy 
print(a)


# In[7]:


# We apply this algorithm to classify our unlabelled tweets

file_path_X = "C:/Users/jonat/Documents/School/NLP/FinalProject/Train_plus_Test/direct_X_FULL.pkl"
X = pickle.load( open(file_path_X, "rb" ) )    
file_path_Y = "C:/Users/jonat/Documents/School/NLP/FinalProject/Train_plus_Test/direct_Y_FULL.pkl"
Y = pickle.load( open(file_path_Y, "rb" ) ) 


# In[8]:


tweets=[]
for i in range(len(X)):
    tweets.append(X[i][0])


# In[9]:


predicted_data_vec=vectorizer.transform(tweets)


# In[10]:


# all data was vectorized but only a specific amount was used as the sentiment vector. This portion is done with the baseline classifiers
z=NB1.unlabelled_prediction(predicted_data_vec)


# In[11]:


# store vectors for sentiment
pickle.dump(z, open('NaiveBayes_labelprediction_last226805.pickle', 'wb'))

