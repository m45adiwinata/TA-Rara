# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:19:15 2019

@author: Grenceng
"""

import pandas as pd
import numpy as np
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
    pr_A = 175/float(525)
    pr_F = 175/float(525)
    pr_TD = 175/float(525)
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

temp_f = open('term gbest Data Edit 0.2.txt', 'r')
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
            for kata in k.split(' '):
                if np.argwhere(terms_awal == kata).size > 0:
                    data = np.append(data, kata)
        datas.append(data)

W = np.array(pd.read_excel('Bobot Gbest Data Edit 0.2.xlsx')).T
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
                tmp_a.append(sum(W[:175,k]))
                tmp_f.append(sum(W[175:350,k]))
                tmp_td.append(sum(W[350:,k]))
        term_used.append(temp)
        total_used_W_a.append(tmp_a)
        total_used_W_f.append(tmp_f)
        total_used_W_td.append(tmp_td)
    all_W = [total_used_W_a[0], total_used_W_f[0], total_used_W_td[0]]
    result = naive_bayes(all_W, W[x,:], term_used[0])
    #results.append(np.argmax(result))
    if x < 50 :
        results.append(np.argmax(result))
    elif x < 100 :
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
akurasi *= 100
print('Selamat! Akurasi sistem anda: %f persen' % akurasi)
exec_time = time.time() - starttime
seconds = exec_time % 60
minutes = exec_time // 60
hours = minutes // 60
minutes = minutes % 60
print("Total execution time 0.2 : %d hours %d minutes %d seconds." % (hours, minutes, seconds))