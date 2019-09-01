
from keras.preprocessing import sequence
import csv
import os
import sklearn
import pickle
from sklearn.utils import shuffle
from keras.layers import Embedding, LSTM, Dense, Dropout
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


from google.colab import drive

# drive.mount('/content/drive/')
drive.mount("/content/drive/", force_remount=True)

# Sentiment140 dataset (for training) load and save
temp=[]
file_path = "/content/drive/My Drive/NLP_DummyTestData/training.1600000.processed.noemoticon.csv"
with open(file_path, encoding = 'latin-1') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    labels=[]
    data=[]
    for row in csv_reader:
      if int(row[0])==0:
        labels.append(0)
      else:
        labels.append(1)
      data.append(row[5])
      line_count += 1



# tokenize with Keras's built in tokenizer
t = Tokenizer()
t.fit_on_texts(data)
vocab_size = len(t.word_index) + 1
print(vocab_size)


# integer encode the documents
encoded_docs = t.texts_to_sequences(data)



maxL=len(max(encoded_docs, key=len))
# pad documents
padded_data = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=maxL, padding='post')


labels_OH = np_utils.to_categorical(labels)


model=Sequential()
model.add(Embedding(vocab_size, 20, input_length=maxL, trainable=False))
model.add(LSTM(20))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))

model.add(Dense(2, activation='softmax'))
ADAM=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss = 'categorical_crossentropy', optimizer=ADAM ,metrics = ['accuracy'])
print(model.summary())




# shuffle all data to generate randomness not native to tweets
padded_data_r, labels_OH_r = shuffle(padded_data, labels_OH, random_state=0)


# testing subset of training points
batch_size=2500
num_epochs=3



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

X_valid, y_valid = padded_data_r[15*100000:16*100000], labels_OH_r[15*100000:16*100000]
X_train2, y_train2 = padded_data_r[:15*100000], labels_OH_r[:15*100000]
M1=model.fit(X_train2, y_train2 , validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs,callbacks=callbacks_list)
M.append(M1)

# load test data

file_path_X = "/content/drive/My Drive/NLP_DummyTestData/direct_X_FULL.pkl"
X = pickle.load( open(file_path_X, "rb" ) )    
file_path_Y = "/content/drive/My Drive/NLP_DummyTestData/direct_Y_FULL.pkl"
Y = pickle.load( open(file_path_Y, "rb" ) ) 

# tokenize/encode tweets
tweets=[]
for i in range(len(X)):
  tweets.append(X[i][0])
  
  # integer encode the documents
encoded_docs_test = t.texts_to_sequences(tweets)

maxL=len(max(encoded_docs, key=len))
# pad documents
padded_data_test = keras.preprocessing.sequence.pad_sequences(encoded_docs_test, maxlen=maxL, padding='post')


# make prediction
prediction=[]
prediction.append(model.predict(padded_data_test))

# convert probability to 0/4 label
final=[]
for i in range(len(prediction[0])):
  if prediction[0][i][1]>0.500000000:
    final.append(0)
  elif prediction[0][i][1]<=0.500000000:
    final.append(4)

# dump prediction
filepath = os.path.join(path, "RNN_trainedonsent140_labelprediction_last226805.pickle") 
with open(filepath, 'wb') as f:
    pickle.dump(final, f)
