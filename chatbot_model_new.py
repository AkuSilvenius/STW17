#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from lxml import etree
import xml.etree.ElementTree as ET
import io
import nltk
import pickle
import gensim
from gensim import corpora, models, similarities
import numpy as np
import os.path
import sPickle
from keras.models import Sequential
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split

print("Loading variables...")
output = open(os.path.join("I:/Python/", "vec_x_data.dat"), "rb")
vec_x = pickle.load(output)
output.close()


output = open(os.path.join("I:/Python/", "vec_y_data.dat"), "rb")
vec_y = pickle.load(output)
output.close()
print("Load complete")
print("to numpy...")
vec_x = np.array(vec_x,dtype=np.float32)
vec_y = np.array(vec_y,dtype=np.float32)
print("done")


x_train,x_test, y_train,y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)
    
model=Sequential()

##model.add(LSTM(kernel_initializer="glorot_normal", return_sequences=True, recurrent_initializer="glorot_normal", input_shape=x_train.shape[1:], units=300, activation="sigmoid"))
##model.add(LSTM(kernel_initializer="glorot_normal", return_sequences=True, recurrent_initializer="glorot_normal", input_shape=x_train.shape[1:], units=300, activation="sigmoid"))
##model.add(LSTM(kernel_initializer="glorot_normal", return_sequences=True, recurrent_initializer="glorot_normal", input_shape=x_train.shape[1:], units=300, activation="sigmoid"))
##model.add(LSTM(kernel_initializer="glorot_normal", return_sequences=True, recurrent_initializer="glorot_normal", input_shape=x_train.shape[1:], units=300, activation="sigmoid"))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM500.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM1000.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM1500.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM2000.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM2500.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM3000.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM3500.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM4000.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM4500.h5');
model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM5000.h5');          
predictions=model.predict(x_test) 
mod = gensim.models.Word2Vec.load('wiki_sg/word2vec.bin'); 
[mod.most_similar([predictions[10][i]])[0] for i in range(15)]




#for testing
#while True:
#    val = input("Give the sentvec: ")
#    print(vec_x[int(val)])


