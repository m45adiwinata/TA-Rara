# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 13:55:39 2019

@author: ACER
"""

import pandas as pd
import numpy as np
import math
import time
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

starttime = time.time()

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

data_path = open('testing data path.txt', 'r')
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
            else:
                if len(term) > 0:
                    token.append(('').join(term))
                    term = []
        token.append(('').join(term))
    return (' ').join(token)

temp_f = open('terms Data Edit.txt', 'r')
terms_awal = []
for f in temp_f:
    f = f.strip()
    terms_awal.append(f)
terms_awal = np.array(terms_awal[0].split(' '))

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
            katas = katastop.split(' ')
            for kata in katas:
                if np.argwhere(terms_awal == kata).size > 0:
                    data = np.append(data, kata)
            #data = np.delete(data, np.argwhere(data == '').flatten())
        datas.append(data)

W = np.array(pd.read_excel('bobot awal Data Edit.xlsx'))
label_training = np.array([])
for i in range(W.shape[0]):
    if i < 175:
        label_training = np.append(label_training, 0)
    elif i >= 175 and i < 350:
        label_training = np.append(label_training, 1)
    else:
        label_training = np.append(label_training, 2)

#pembobotan
terms = terms_awal

tf = np.zeros((terms.size, len(cerpens[0])+len(cerpens[1])+len(cerpens[2])))
for i in range(len(datas)):
    for j in range(len(datas[i])):
        for k in range(tf.shape[0]):
            if terms[k] == datas[i][j]:
                tf[k][i] += 1

IDF = np.array([])
for i in range(terms.size):
    D = len(datas)
    df = len(np.nonzero(tf[i,:])[0])
    if df > 0:
        IDF = np.append(IDF, math.log(D/df))
    else:
        IDF = np.append(IDF, 0)

W_testing = np.zeros((len(datas), terms.size))
for i in range(W_testing.shape[0]):
    for j in range(W_testing.shape[1]):
        W_testing[i,j] = tf[j,i] * (IDF[j] + 1)

label_testing = np.array([])
for i in range(len(datas)):
    if i < 50:
        label_testing = np.append(label_testing, 0)
    elif i >= 50 and i < 100:
        label_testing = np.append(label_testing, 1)
    else:
        label_testing = np.append(label_testing, 2)

gnb = GaussianNB()
gnb.fit(W, label_training)
pred = gnb.predict(W_testing)

print('Selamat! Akurasi sistem anda: %f' % metrics.accuracy_score(pred, label_testing))
exec_time = time.time() - starttime
seconds = exec_time % 60
minutes = exec_time // 60
hours = minutes // 60
minutes = minutes % 60
print("Total execution time : %d hours %d minutes %d seconds." % (hours, minutes, seconds))
