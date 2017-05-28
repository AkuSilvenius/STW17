import os
from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model

model = load_model('best_model_val_acc.hdf5')
mod = gensim.models.Word2Vec.load('wiki_sg/word2vec.bin');

while True:
##    wrd = input("new: ")
##    print(mod.most_similar(wrd))
    msg = input("Me: ")
    sentence = nltk.word_tokenize(msg.lower())
    sentvec = [mod[w] for w in sentence if w in mod.vocab]
    sentvec[14:]=[]
    #print(sentvec)
    
    sentend = np.ones(300,dtype=np.float32)
    sentvec.append(sentend)
    for i in range(15-len(sentvec)):
        sentvec.append(sentend)
    sentvec=np.array([sentvec])
    predictions = model.predict(sentvec)
    #print(predictions)
    outputlist = []
    for i in range(15):
        word = mod.most_similar([predictions[0][i]])[0][0]
        #print(word)
        outputlist.append(word)
    output = ' '.join(outputlist)
    print("Bot :", output)
