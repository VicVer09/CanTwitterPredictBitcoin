
from keras.preprocessing import sequence
import csv
import os
import sklearn
import pickle
from keras.layers import Embedding, LSTM, Dense, Dropout,Bidirectional

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import numpy as np
from sklearn.decomposition import PCA
from keras.preprocessing.text import text_to_word_sequence
import glob
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.layers import LeakyReLU
from keras.optimizers import Adam, Nadam
import keras
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import random


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# the next few are required to use Google Collab, which we did for the GPU
from google.colab import drive

# drive.mount('/content/drive/')
drive.mount("/content/drive/", force_remount=True)

# train data load and save data

file_path_X = "/content/drive/My Drive/NLP_DummyTestData/direct_X_FULL.pkl"
X = pickle.load( open(file_path_X, "rb" ) )    
file_path_Y = "/content/drive/My Drive/NLP_DummyTestData/direct_Y_FULL.pkl"
Y = pickle.load( open(file_path_Y, "rb" ) ) 


# recover tweets and candles
tweets=[]
for i in range(len(X)):
  tweets.append(X[i][0])

candle_1=[]
candle_5=[]
candle_60=[]
for i in range(len(Y)):
  if Y[i][0]>=0:
    candle_1.append(0) 
  if Y[i][0]<0:
    candle_1.append(1)
  if Y[i][1]>=0:
    candle_5.append(0)
  if Y[i][1]<0:   
    candle_5.append(1)
  if Y[i][2]>=0:  
    candle_60.append(0)
  if Y[i][2]<0:    
    candle_60.append(1) 


# Tokenize text with Keras's built in tokenizer
# t = Tokenizer(num_words=998)
t = Tokenizer()
t.fit_on_texts(tweets)
vocab_size = len(t.word_index) + 1
print(vocab_size)


# integer encode the documents
encoded_docs = t.texts_to_sequences(tweets)



maxL=len(max(encoded_docs, key=len))
trunc=10
# pad documents
padded_data = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=maxL, padding='post')


# One-hot encoding of labels
labels_1 = np_utils.to_categorical(candle_1)
labels_5 = np_utils.to_categorical(candle_5)
labels_60 = np_utils.to_categorical(candle_60)




# Model definition
model=Sequential()
model.add(Embedding(vocab_size, 20, input_length=maxL, trainable=False))
model.add(LSTM(20))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))
ADAM=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss = 'categorical_crossentropy', optimizer=ADAM ,metrics = ['accuracy'])
print(model.summary())

# parameter setting
batch_size=1000
num_epochs = 1
M=[]
path='/content/drive/My Drive/NLP_DummyTestData/'
filepath = os.path.join(path, "weights{epoch:02d}-{loss:.4f}-Word2Vec.h5")  
callbacks_list = [  keras.callbacks.EarlyStopping(
        monitor='val_acc', 
        patience= 10,
        mode='max',
        verbose=1),
    keras.callbacks.ModelCheckpoint(filepath,
        monitor='val_acc', 
        save_best_only=True, 
        mode='max',
        verbose=0)
]

# run model - done for final grid search parameters
X_valid, y_valid = padded_data[400000:500000], labels_1[400000:500000]
X_train2, y_train2 = padded_data[:400000], labels_1[:400000]
M1=model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs,callbacks=callbacks_list)
M.append(M1)


# this code was used by grid search for number of LSTM layers units, number of Dense layer units, uni or bidirectional LSTM and learning rate

def create_model(hidden_units_LSTM):
  print("I m here",hidden_units_LSTM)
  model=Sequential()
  model.add(Embedding(vocab_size, 10, input_length=maxL, trainable=False))
  model.add(LSTM(hidden_units_LSTM))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  ADAM=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  model.compile(loss = 'categorical_crossentropy', optimizer=ADAM ,metrics = ['accuracy'])
  return model

# grid search code

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# create model
model_GS = KerasClassifier(build_fn=create_model, epochs=7, batch_size=1000, verbose=1)
# define the grid search parameters
hidden_units_LSTM = [1 , 10, 20, 50, 100]
param_grid = dict(hidden_units_LSTM=hidden_units_LSTM)
grid = GridSearchCV(estimator=model_GS, param_grid=param_grid, n_jobs=-1, cv=2)
grid_result = grid.fit(padded_data, labels_1)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print("%f (%f) with: %r" % (mean, stdev, param)) 




# to predict unlabelled data 
prediction=[]

prediction.append(model.predict(padded_data))

# converts prediction from probability to labels 0/4 (similar to NB and VADER)
final=[]
for i in range(len(prediction[0])):
  if prediction[0][i][1]>0.500000000:
    final.append(0)
  elif prediction[0][i][1]<=0.500000000:
    final.append(4)



# Dump sentiment information
filepath = os.path.join(path, "RNN_trainedonBIT_labelprediction_last226805.pickle") 
with open(filepath, 'wb') as f:
    pickle.dump(final, f)
