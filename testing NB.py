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

def naive_bayes(test, W, W_mean, W_var, W_std):
    idx = np.flatnonzero(test)
    p = np.ones(3)
    for i in range(p.size):
        for index in idx:
            p[i] *= 1/(math.sqrt(2*math.pi)*W_std[i][index])*math.e**((W[index]-W_mean[i][index])**2/(2*W_var[i][index]))
    return p

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
            katadasar = stemmer.stem(k)
            katas = katadasar.split(' ')
            for kata in katas:
                if np.argwhere(terms_awal == kata).size > 0:
                    data = np.append(data, kata)
        datas.append(data)

W = np.array(pd.read_excel('bobot awal Data Edit.xlsx')).T
label_training = np.array([])
for i in range(W.shape[0]):
    if i < 175:
        label_training = np.append(label_training, 0)
    elif i >= 175 and i < 350:
        label_training = np.append(label_training, 1)
    else:
        label_training = np.append(label_training, 2)

W_anak, W_fantasi, W_tidik = W[:175], W[175:350], W[350:]
W_anak_term_mean, W_fantasi_term_mean, W_tidik_term_mean = np.array([]), np.array([]), np.array([])
W_anak_term_variance, W_fantasi_term_variance, W_tidik_term_variance = np.array([]), np.array([]), np.array([])
W_anak_term_stdev, W_fantasi_term_stdev, W_tidik_term_stdev = np.array([]), np.array([]), np.array([])
for i in range(W_anak.shape[1]):
    W_anak_term_mean = np.append(W_anak_term_mean, np.mean(W_anak[:,i]))
    W_anak_term_variance = np.append(W_anak_term_variance, np.var(W_anak[:,i]))
    W_anak_term_stdev = np.append(W_anak_term_stdev, np.std(W_anak[:,i]))
for i in range(W_fantasi.shape[1]):
    W_fantasi_term_mean = np.append(W_fantasi_term_mean, np.mean(W_fantasi[:,i]))
    W_fantasi_term_variance = np.append(W_fantasi_term_variance, np.var(W_fantasi[:,i]))
    W_fantasi_term_stdev = np.append(W_fantasi_term_stdev, np.std(W_fantasi[:,i]))
for i in range(W_tidik.shape[1]):
    W_tidik_term_mean = np.append(W_tidik_term_mean, np.mean(W_tidik[:,i]))
    W_tidik_term_variance = np.append(W_tidik_term_variance, np.var(W_tidik[:,i]))
    W_tidik_term_stdev = np.append(W_tidik_term_stdev, np.std(W_tidik[:,i]))

#pembobotan
terms = terms_awal
test_set = np.zeros((len(datas), terms.size))
tf = np.zeros((terms.size, len(datas)))
for i in range(len(datas)):
    for j in range(len(datas[i])):
        for k in range(terms.size):
            if np.argwhere(datas[i] == terms[k]).size > 0:
                test_set[i][k] == 1
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

pred = np.array([])
i = 0
for test in test_set:
    W_mean = [W_anak_term_mean, W_fantasi_term_mean, W_tidik_term_mean]
    W_var = [W_anak_term_variance, W_fantasi_term_variance, W_tidik_term_variance]
    W_stdev = [W_anak_term_stdev, W_fantasi_term_stdev, W_tidik_term_stdev]
    P = naive_bayes(test, W_testing[i], W_mean, W_var, W_stdev)
    pred = np.append(pred, np.argmax(P))
    i += 1
    
    
print('Selamat! Akurasi sistem anda 1.0 : %f' % metrics.accuracy_score(pred, label_testing))
exec_time = time.time() - starttime
seconds = exec_time % 60
minutes = exec_time // 60
hours = minutes // 60
minutes = minutes % 60
print("Total execution time 1.0 : %d hours %d minutes %d seconds." % (hours, minutes, seconds))
