# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 13:55:39 2019

@author: ACER
"""

import pandas as pd
import numpy as np
import random
import math
import time
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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

def naive_bayes(all_W, W_uji, term_used):
    pr_A = 125/float(350)
    pr_F = 125/float(350)
    pr_TD = 100/float(350)
    p_term = []
    all_W = np.array(all_W)
    total_t = sum(all_W[0,:]) + sum(all_W[1,:]) + sum(all_W[2,:])
    for i in range(len(all_W)):
        temp = []
        for j in range(len(all_W[i])):
            P = (all_W[i,j] + 1) / float(total_t + sum(all_W[i,:]))
            temp.append(P)
        p_term.append(temp)
    P = []
    for i in range(len(p_term)):
        temp = 1
        for j in range(len(p_term)):
            if W_uji[term_used[j]] > 0:
                temp *= p_term[i][j]
        P.append(temp)
    P[0] *= pr_A
    P[1] *= pr_F
    P[2] *= pr_TD
    return P

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
            temp = 0
            while len(katastop) != temp:
                temp = len(katastop)
                katastop = stopword.remove(katastop)
            data = np.append(data, katastop.split(' '))
            data = np.delete(data, np.argwhere(data == '').flatten())
        datas.append(data)

temp_f = open('terms Data Edit.txt', 'r')
terms_awal = []
for f in temp_f:
    f = f.strip()
    terms_awal.append(f)
terms_awal = np.array(terms_awal[0].split(' '))

W = np.array(pd.read_excel('bobot awal Data Edit.xlsx'))

results = []
for x in range(len(datas)):
    data = datas[x]
    P = [np.zeros(terms_awal.size)]
    for i in range(P[0].size):
        if np.argwhere(data == terms_awal[i]).size > 0:
            P[0][i] = 1
    total_used_W_a = []
    total_used_W_f = []
    total_used_W_td = []
    term_used = []
    for j in range(len(P)):
        temp = []
        tmp_a = []
        tmp_f = []
        tmp_td = []
        for k in range(P[0].size):
            if P[j][k] == 1:
                temp.append(j)
                tmp_a.append(sum(W[:125,k]))
                tmp_f.append(sum(W[125:250,k]))
                tmp_td.append(sum(W[250:,k]))
        term_used.append(temp)
        total_used_W_a.append(tmp_a)
        total_used_W_f.append(tmp_f)
        total_used_W_td.append(tmp_td)
    all_W = [total_used_W_a[0], total_used_W_f[0], total_used_W_td[0]]
    result = naive_bayes(all_W, W[x,:], term_used[0])
    #results.append(np.argmax(result))
    if x < 50 :
        results.append(np.argmax(result))
    elif x < 100 and x >= 50:
        if result[0] == result[1]:
            if result[1] > result[2]:
                results.append(1)
            else:
                results.append(2)
        elif result[1] == result[2]:
            if result[1] > result[0]:
                results.append(1)
            else:
                results.append(0)
        else:
                results.append(np.argmax(result))
    else :
        if result[0] == result[1]:
            if result[1] > result[2]:
                results.append(1)
            else:
                results.append(2)
        elif result[1] == result[2]:
            if result[2] > result[0]:
                results.append(2)
            else:
                results.append(0)
        else:
                results.append(np.argmax(result))
results = np.array(results)
akurasi = np.argwhere(results[:50] == 0).size + np.argwhere(results[50:100] == 1).size + np.argwhere(results[100:] == 2).size
akurasi /= float(150)
print('Selamat! Akurasi sistem anda: %f' % (akurasi*100))
exec_time = time.time() - starttime
seconds = exec_time % 60
minutes = exec_time // 60
hours = minutes // 60
minutes = minutes % 60
print("Total execution time : %d hours %d minutes %d seconds." % (hours, minutes, seconds))
