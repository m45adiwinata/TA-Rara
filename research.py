# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 01:09:02 2019

@author: Grenceng
"""

import os
import numpy as np
import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

data_path = open('data path.txt', 'r')
path = []
for p in data_path:
    p = p.strip()
    path.append(p)
cerpens = []
for p in path:
    cerpens.append([os.path.join(p,fname) for fname in os.listdir(p) if fname.endswith('.txt')])

def tokenize(kalimat):
    words = kalimat.split(' ')
    token = []
    for word in words:
        lw = list(word)
        term = []
        for w in lw:
            if w.isalpha() == True:
                term.append(w)
        token.append(('').join(term))
    return (' ').join(token)
    
datas = []
for cerpen in cerpens:
    for cer in cerpen:
        contents = open(cer, 'r')
        kalimat = []
        for c in contents:
            c = c.strip()
            kalimat.append(c)
        data = np.array([])
        for k in kalimat:
            katastop = stopword.remove(k)
            katastop = tokenize(katastop)
            katadasar = stemmer.stem(katastop)
            katastop = stopword.remove(katadasar)
            data = np.append(data, katastop.split(' '))
        datas.append(data)

#pembobotan
terms = np.array([])
for i in range(len(datas)):
    for j in range(datas[i].size):
        if terms.size == 0:
            terms = np.append(terms, datas[i][j])
        else:
            if np.argwhere(terms == datas[i][j]).size == 0:
                terms = np.append(terms, datas[i][j])

tf = np.zeros((terms.size, 350))
for i in range(len(datas)):
    for j in range(len(datas[i])):
        for k in range(tf.shape[0]):
            if terms[k] == datas[i][j]:
                tf[k][i] += 1

IDF = np.array([])
for i in range(terms.size):
    D = len(datas)
    df = len(np.nonzero(tf[i,:])[0])
    IDF = np.append(IDF, math.log(D/df))

W = np.zeros((len(datas), terms.size))
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        W[i,j] = tf[j,i] * (IDF[j] + 1)
df = pd.DataFrame(W)
df.to_excel('bobot awal.xlsx', index=False)
tf = pd.DataFrame(tf)
tf.to_excel('term frequency.xlsx', index=False, header=False)
file = open('terms.txt', 'w')
for t in terms:
    file.write("%s " % t)
file.close()
idf = pd.DataFrame(IDF)
idf.to_excel('IDF.xlsx', index=False, header=False)
