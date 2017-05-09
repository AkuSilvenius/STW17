#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from lxml import etree
import xml.etree.ElementTree as ET
import io
import nltk
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim import corpora, models, similarities
import numpy as np
import os.path

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

for i in range(0,50):
    print("x[",i,"]: ", x[i])
    print("y[",i,"]: ", y[i])
    print("")


# Jakaa lauseen sanoihin
# esim.  [No, sir.] -> ['no', ',', 'sir', '.']
tokenized_x = []
tokenized_y = []

print("Tokenizing...")
for i in range(len(x)):
    tokenized_x.append(nltk.word_tokenize(x[i].lower()))
    tokenized_y.append(nltk.word_tokenize(y[i].lower()))

print("Tokenization completed")
print(tokenized_x[10], tokenized_y[10])
print(tokenized_x[11], tokenized_y[12])

print("Dumping data")
output = open(os.path.join("I:/Python/", "tokenized_data.dat"), "wb")
pickle.dump([tokenized_x, tokenized_y], output)
#tok_x, tok_y = pickle.load(output)
output.close()
print("Dump complete")

