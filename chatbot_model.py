#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import print_function
#from lxml import etree
#import xml.etree.ElementTree as ET
#import io
#import nltk
import pickle
import gensim
from gensim import corpora, models, similarities
import numpy as np
import os.path
#import sPickle
from keras.models import Sequential
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model


model = gensim.models.Word2Vec.load('wiki_sg/word2vec.bin');

print("Loading data...")
output = open(os.path.join("I:/Python/", "tokenized_data.dat"), "rb")
tokenized_x, tokenized_y = pickle.load(output)
output.close()
print("Load complete")
print("Modifying data...")
sentend = np.ones(300,dtype=np.float32)

vec_x = []
vec_y = []
i = 1
for sent in tokenized_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)
    i = i +1
    if i > 10000:
        break
#print("lauseiden määrä: ", i)
print(len(vec_x))
i = 1
for sent in tokenized_y:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_y.append(sentvec)
    i = i +1
    if i > 10000:
        break

for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)    
            
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)

print("Done.")

print("Creating vec_x...")
vec_x = np.array(vec_x,dtype=np.float32)
print("Done.")
#print("Dumping vec_x...")
#output = open(os.path.join("I:/Python/", "vec_x_data.dat"), "wb")
#pickle.dump(vec_x, output)
##tok_x, tok_y = pickle.load(output)
#output.close()
#print("Dump complete")
#vec_x = None

print("Creating vec_y...")
vec_y = np.array(vec_y,dtype=np.float32)
print("Done.")

#print("Dumping data...")
#output = open(os.path.join("I:/Python/", "vec_y_data.dat"), "wb")
#pickle.dump(vec_y, output)
##tok_x, tok_y = pickle.load(output)
#output.close()
#print("Dump complete")
#vec_y = None


##for i in range(10):
##    predictions = [vec_x[i+70]]
##    #print(predictions)
##    #print(predictions)
##    outputlist = []
##    for i in range(15):
##        word = model.most_similar([predictions[0][i]])[0][0]
##        #print(word)
##        outputlist.append(word)
##    output = ' '.join(outputlist)
##    print("Bot :", output)


    
x_train,x_test, y_train,y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)
checkpointer = ModelCheckpoint(filepath="best_model_val_acc.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
checkpointer2 = ModelCheckpoint(filepath="best_model_acc.hdf5", monitor='acc', verbose=1, save_best_only=True)

dropout = 0.1

##model=Sequential()
##
##opt =  optimizers.SGD(lr=0.01)
##model.add(LSTM(kernel_initializer="glorot_normal", return_sequences=True, recurrent_initializer="glorot_normal", input_shape=x_train.shape[1:], units=300, activation="relu"))
##model.add(BatchNormalization())
##model.add(Dropout(dropout))
##
##model.add(LSTM(kernel_initializer="glorot_normal", return_sequences=True, recurrent_initializer="glorot_normal", input_shape=x_train.shape[1:], units=300, activation="relu"))
##model.add(BatchNormalization())
##model.add(Dropout(dropout))
##
##model.add(LSTM(kernel_initializer="glorot_normal", return_sequences=True, recurrent_initializer="glorot_normal", input_shape=x_train.shape[1:], units=300, activation="relu"))
##model.add(BatchNormalization())
##model.add(Dropout(dropout))
##
##model.add(LSTM(kernel_initializer="glorot_normal", return_sequences=True, recurrent_initializer="glorot_normal", input_shape=x_train.shape[1:], units=300, activation="relu"))
##model.add(BatchNormalization())
###model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
###model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
###model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
###model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
##model.compile(loss='cosine_proximity', optimizer="adam", metrics=['accuracy'])


model = load_model('LSTM10000.h5')

b_size = 124
epoch_size = 10
generations = 20


for i in range(1,generations+1):
    print("i: ", i)
    model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test), verbose=2, callbacks=[checkpointer, checkpointer2]) #epoch originally 500
    model.save('LSTM'+str(10000+500*i)+'.h5');
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (score[1]*100))
#model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM1000.h5');
##model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM1500.h5');
##model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM2000.h5');
##model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM2500.h5');
##model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM3000.h5');
##model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM3500.h5');
##model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM4000.h5');
##model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM4500.h5');
##model.fit(x_train, y_train, batch_size=b_size, epochs=epoch_size, validation_data=(x_test, y_test))
##model.save('LSTM5000.h5');          
predictions=model.predict(x_test) 
mod = gensim.models.Word2Vec.load('wiki_sg/word2vec.bin');
[mod.most_similar([predictions[10][i]])[0] for i in range(15)]



#for testing
#while True:
#    val = input("Give the sentvec: ")
#    print(vec_x[int(val)])


