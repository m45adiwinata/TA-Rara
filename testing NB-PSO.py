# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:19:15 2019

@author: Grenceng
"""

import pandas as pd
import numpy as np
import random
import math
import time
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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
        datas.append(data)

terms = np.array([])
for i in range(len(datas)):
    for j in range(datas[i].size):
        if terms.size == 0:
            terms = np.append(terms, datas[i][j])
        else:
            if np.argwhere(terms == datas[i][j]).size == 0:
                terms = np.append(terms, datas[i][j])
                
learned_terms = np.array([])
file = open('term gbest.txt', 'r')
for f in file:
    f = f.strip()
    learned_terms = np.append(learned_terms, f.split(' '))

bobot = np.array(pd.read_excel('Bobot Gbest.xlsx'))
results = []
for data in datas:
    total_used_W_a = []
    total_used_W_f = []
    total_used_W_td = []
    term_used = []
    for i in range(learned_terms.size):
        if np.argwhere(learned_terms[i] == data).size > 0:
            total_used_W_a.append(np.sum(bobot[i,:125]))
            total_used_W_f.append(np.sum(bobot[i,125:250]))
            total_used_W_td.append(np.sum(bobot[i,250:]))
            term_used.append(i)
    all_W = [total_used_W_a, total_used_W_f, total_used_W_td]
    P = naive_bayes(all_W, )