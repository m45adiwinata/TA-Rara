# -*- coding: uW-8 -*-
"""
Created on Fri May 24 23:41:22 2019

@author: ACER
"""

import pandas as pd
import numpy as np
import random
import math

W = np.reshape(np.array(pd.read_excel('bobot awal.xlsx')), (350, -1))
#terms = np.array(pd.read_excel('terms.xlsx'))
file = open('terms.txt', 'r')
for t in file:
    t = t.strip()
    terms = t.split(' ')
file.close()

bobot_awal = np.array(pd.read_excel('bobot awal.xlsx'))
IDF = np.array(pd.read_excel('IDF.xlsx'))

def naive_bayes(all_W, W_uji, term_used):
    pr_A = 175/float(500)
    pr_F = 175/float(500)
    pr_TD = 150/float(500)
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

def hitung_fitness(alpha, beta, total_W_a, total_W_f, total_W_td, populasi):
    fitness = []
    for i in range(populasi.shape[0]):
        all_W = [total_W_a[i], total_W_f[i], total_W_td[i]]
        result = []
        for j in range(125):
            P = naive_bayes(all_W, W[:,j], term_used[i])
            result.append(np.argmax(P))
        for j in range(175, 300):
            P = naive_bayes(all_W, W[:,j], term_used[i])
            result.append(np.argmax(P))
        for j in range(350, 450):
            P = naive_bayes(all_W, W[:,j], term_used[i])
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
            Fmeasures.append(2 * recalls[j] * precisions[j] / (recalls[j] + precisions[j]))
        fit = alpha * np.mean(Fmeasures) + beta * (len(terms) - len(term_used[i])) / float(len(terms))
        fitness.append(fit)
    return fitness

b_inersia = 0.6
c1 = 0.5
c2 = 0.5
#INISIASI POPULASI
populasi = np.zeros((30,len(terms)-1))
for i in range(populasi.shape[0]):
    for j in range(populasi.shape[1]):
        populasi[i,j] = random.randint(0,1)
term_used = []
total_W_a = []
total_W_f = []
total_W_td = []
for i in range(populasi.shape[0]):
    temp = []
    tmp_a = []
    tmp_f = []
    tmp_td = []
    for j in range(populasi.shape[1]):
        if populasi[i,j] == 1:
            temp.append(j)
            tmp_a.append(sum(W[j,:175]))
            tmp_f.append(sum(W[j,175:350]))
            tmp_td.append(sum(W[j,350:]))
    term_used.append(temp)
    total_W_a.append(tmp_a)
    total_W_f.append(tmp_f)
    total_W_td.append(tmp_td)
#HITUNG FITNESS
alpha = 0.85
beta = 0.15
v = np.ones((populasi.shape))
for i in range(2):
    fitness = hitung_fitness(alpha, beta, total_W_a, total_W_f, total_W_td, populasi)
    if i == 0:
        pbest_val = fitness
        pbest_iter_idx = np.zeros(populasi.shape[0])
    else:
        new_pbest = fitness
        for j in range(len(pbest_val)):
            if new_pbest[j] > pbest_val[j]:
                pbest_val[j] = new_pbest[j]
                pbest_iter_idx[j] = i
    gbest_idx = np.argmax(fitness)
    gbest_val = fitness[np.argmax(fitness)]
    for j in range(v.shape[0]):
        for k in range(v.shape[1]):
            v[j,k] = b_inersia * v + c1 * random.randint(0,1) * (pbest_val[j] - populasi[j,k]) + c2 * random.randint(0,1) * (gbest_val - populasi[j,k])
            sig_v = 1 / (1 + pow(math.e, (-v[j,k])))
            if populasi[j,k] < sig_v:
                populasi[j,k] = 1
            else:
                populasi[j,k] = 0
    
