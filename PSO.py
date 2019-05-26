# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:41:22 2019

@author: ACER
"""

import pandas as pd
import numpy as np
import random

tf = np.array(pd.read_excel('term frequency.xlsx'))
#terms = np.array(pd.read_excel('terms.xlsx'))
file = open('terms.txt', 'r')
for t in file:
    t = t.strip()
    terms = t.split(' ')
file.close()

bobot_awal = np.array(pd.read_excel('bobot awal.xlsx'))
IDF = np.array(pd.read_excel('IDF.xlsx'))

def naive_bayes(all_tf, tf_uji, term_used):
    pr_A = 175/float(500)
    pr_F = 175/float(500)
    pr_TD = 150/float(500)
    p_term = []
    all_tf = np.array(all_tf)
    total_t = sum(all_tf[0,:]) + sum(all_tf[1,:]) + sum(all_tf[2,:])
    for i in range(len(all_tf)):
        temp = []
        for j in range(len(all_tf[i])):
            P = (all_tf[i,j] + 1) / float(total_t + sum(all_tf[i,:]))
            temp.append(P)
        p_term.append(temp)
    P = []
    for i in range(len(p_term)):
        temp = 1
        for j in range(len(p_term)):
            if tf_uji[term_used[j]] > 0:
                temp *= p_term[i][j]
        P.append(temp)
    P[0] *= pr_A
    P[1] *= pr_F
    P[2] *= pr_TD
    return P

b_inersia = 0.6
c1 = 0.5
c2 = 0.5
#INISIASI POPULASI
populasi = np.zeros((30,len(terms)-1))
for i in range(populasi.shape[0]):
    for j in range(populasi.shape[1]):
        populasi[i,j] = random.randint(0,1)
term_used = []
total_tf_a = []
total_tf_f = []
total_tf_td = []
for i in range(populasi.shape[0]):
    temp = []
    tmp_a = []
    tmp_f = []
    tmp_td = []
    for j in range(populasi.shape[1]):
        if populasi[i,j] == 1:
            temp.append(j)
            tmp_a.append(sum(tf[j,:175]))
            tmp_f.append(sum(tf[j,175:350]))
            tmp_td.append(sum(tf[j,350:]))
    term_used.append(temp)
    total_tf_a.append(tmp_a)
    total_tf_f.append(tmp_f)
    total_tf_td.append(tmp_td)
#HITUNG FITNESS
alpha = 0.85
beta = 0.15

fitness = []
for i in range(populasi.shape[0]):
    all_tf = [total_tf_a[i], total_tf_f[i], total_tf_td[i]]
    result = []
    for j in range(125):
        P = naive_bayes(all_tf, tf[:,j], term_used[i])
        result.append(np.argmax(P))
    for j in range(175, 300):
        P = naive_bayes(all_tf, tf[:,j], term_used[i])
        result.append(np.argmax(P))
    for j in range(350, 450):
        P = naive_bayes(all_tf, tf[:,j], term_used[i])
        result.append(np.argmax(P))
    result = np.array(result)
    if np.argwhere(result == 0).size > 0:
        precisions = [np.argwhere(result[:150] == 0).size/float(np.argwhere(result == 0).size)]
    else:
        precisions = [0]
    if np.argwhere(result == 1).size > 0:
        precisions.append(np.argwhere(result[150:300] == 1).size/float(np.argwhere(result == 1).size))
    else:
        precisions.append(0)
    if np.argwhere(result == 2).size > 0:
        precisions.append(np.argwhere(result[300:] == 2).size/float(np.argwhere(result == 2).size))
    else:
        precisions.append(0)
    recalls = [np.argwhere(result[:150] == 0).size/float(150)]
    recalls.append(np.argwhere(result[150:300] == 1).size/float(150))
    recalls.append(np.argwhere(result[300:] == 2).size/float(100))
    Fmeasures = []
    for j in range(3):
        Fmeasures.append(2 * recalls[i] * precisions[i] / (recalls[i] + precisions[i]))
    fit = alpha * np.mean(Fmeasures) + beta * (len(terms) - len(term_used[i])) / float(len(terms))
    fitness.append(fit)
    break
