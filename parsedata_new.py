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


model = gensim.models.Word2Vec.load('wiki_sg/word2vec.bin');
parser = etree.XMLParser(recover=True)

tree = ET.parse('./data/MovieDiC_V2.xml', parser=parser)
root = tree.getroot()
file = io.open('cleanedData.txt','w',encoding='utf-8')

iterator = tree.getiterator()
data = []
print('Cleaning up the data...')
for m in iterator:
    #print(m.tag)
    for di in m.findall('dialogue'):
        dialog = []
        #print("dialogue id: " + di.get('id'))
        for u in di.findall('utterance'):
            sentence = u.text + u"\n"
            dialog.append(sentence)
            #file.write((u.text).join(["\n"]))
    #print("dialogue changes")
        data.append(dialog)
file.close()
print('Clean up complete!')
#print(data[122])


x = []
y = []
for i in range(len(data)):
    for j in range(len(data[i])):
        if j<len(data[i])-1:
            x.append(data[i][j])
            y.append(data[i][j+1])
print(x[10], y[10])
print(x[11], y[11])

# Jakaa lauseen sanoihin
# esim.  [No, sir.] -> ['no', ',', 'sir', '.']
tokenized_x = []
tokenized_y = []

for i in range(len(x)):
    tokenized_x.append(nltk.word_tokenize(x[i].lower()))
    tokenized_y.append(nltk.word_tokenize(y[i].lower()))


print(tokenized_x[10], tokenized_y[10])
print(tokenized_x[11], tokenized_y[12])

sentend = np.ones(300,dtype=np.float32)


# Mitä tää tekee?? ->>
vec_x = []
for sent in tokenized_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)

while True:
    val = input("Give the sentvec: ")
    print(vec_x[int(val)])
    
#output = open("data.dat", "wb")
#pickle.dump(data, output)
#array = pickle.load(output)
