# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:41:22 2019

@author: ACER
"""

import pandas as pd
import numpy as np
import random
import math
import time

starttime = time.time()
W = np.array(pd.read_excel('bobot awal Data Edit.xlsx')).T
#terms = np.array(pd.read_excel('terms.xlsx'))
file = open('terms Data Edit.txt', 'r')
for t in file:
    t = t.strip()
    terms = t.split(' ')
file.close()
#IDF = np.array(pd.read_excel('IDF.xlsx'))

def naive_bayes(all_W, W_uji):
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
        for j in range(len(p_term[i])):
            if W_uji[j] > 0:
                temp *= p_term[i][j]
        P.append(temp)
    P[0] *= pr_A
    P[1] *= pr_F
    P[2] *= pr_TD
    return P

def hitung_fitness(total_W_a, total_W_f, total_W_td, populasi, alpha, beta):
    fitness = []
    for i in range(populasi.shape[0]):
        all_W = [total_W_a[i], total_W_f[i], total_W_td[i]]
        result = []
        print("naive bayes clasification step")
        for j in range(175):
            P = naive_bayes(all_W, populasi[i])
            result.append(np.argmax(P))
        for j in range(175, 350):
            P = naive_bayes(all_W, populasi[i])
            if P[0] == P[1]:
                if P[1] > P[2]:
                    result.append(1)
                else:
                    result.append(2)
            elif P[1] == P[2]:
                if P[1] > P[0]:
                    result.append(1)
                else:
                    result.append(0)
            else:
                    result.append(np.argmax(P))
        for j in range(350, W.shape[1]):
            P = naive_bayes(all_W, populasi[i])
            if P[0] == P[1]:
                if P[1] > P[2]:
                    result.append(1)
                else:
                    result.append(2)
            elif P[1] == P[2]:
                if P[2] > P[0]:
                    result.append(2)
                else:
                    result.append(0)
            else:
                    result.append(np.argmax(P))
        result = np.array(result)
        
        print("calculate precisions")
        if np.argwhere(result == 0).size > 0:
            precisions = [np.argwhere(result[:175] == 0).size/float(np.argwhere(result == 0).size)]
        else:
            precisions = [0]
        if np.argwhere(result == 1).size > 0:
            precisions.append(np.argwhere(result[175:350] == 1).size/float(np.argwhere(result == 1).size))
        else:
            precisions.append(0)
        if np.argwhere(result == 2).size > 0:
            precisions.append(np.argwhere(result[350:] == 2).size/float(np.argwhere(result == 2).size))
        else:
            precisions.append(0)
            
        print("calculate recalls")
        recalls = [np.argwhere(result[:175] == 0).size/float(175)]
        recalls.append(np.argwhere(result[175:350] == 1).size/float(175))
        recalls.append(np.argwhere(result[350:] == 2).size/float(175))
        
        print("determining fmeasures")
        Fmeasures = []
        for j in range(3):
            if recalls[j]+precisions[j] > 0:
                Fmeasures.append(2 * recalls[j] * precisions[j] / float((recalls[j] + precisions[j])))
            else:
                Fmeasures.append(0)
        F1 = np.mean(Fmeasures)
        N = len(terms)
        F = len(term_used[i])
        fit = (alpha*F1)+(beta*((N-F)/N))
        fitness.append(fit)
    print("Fitness : ", fitness)    
    return fitness

b_inersia = 0.1
c1 = 2
c2 = 2
alpha = 0.85
beta = 0.15
#INISIASI POPULASI
print("initiating population")
populasi = np.zeros((1,len(terms)))
for i in range(populasi.shape[0]):
    for j in range(populasi.shape[1]):
        populasi[i,j] = random.randint(0,1)

#HITUNG FITNESS
print("evaluate fitness and generating")
v = np.zeros((populasi.shape))
gbest_values = np.array([])
gbest_val = 0
for i in range(30):
    print("generation", (i+1))
    print("find and sum used features")
    gbest_conv = 0
    term_used = []
    total_W_a = []
    total_W_f = []
    total_W_td = []
    for j in range(populasi.shape[0]):
        temp = []
        tmp_a = []
        tmp_f = []
        tmp_td = []
        for k in range(populasi.shape[1]):
            if populasi[j,k] == 1:
                temp.append(k)
                tmp_a.append(sum(W[k,:175]))
                tmp_f.append(sum(W[k,175:350]))
                tmp_td.append(sum(W[k,350:]))
        term_used.append(temp)
        total_W_a.append(tmp_a)
        total_W_f.append(tmp_f)
        total_W_td.append(tmp_td)
        
    fitness = hitung_fitness(total_W_a, total_W_f, total_W_td, populasi, alpha, beta)
    
    if i == 0:
        pbest_val = fitness
        pbest_iter_idx = np.zeros(populasi.shape[0])
        pbest_pop = populasi
    else:
        new_pbest = fitness
        for j in range(len(pbest_val)):
            if new_pbest[j] > pbest_val[j]:
                pbest_val[j] = new_pbest[j]
                pbest_iter_idx[j] = i
                pbest_pop[j] = populasi[j]
    
    gbest_idx = np.argmax(pbest_val)
    
    for j in range(len(pbest_val)):
        if pbest_val[j] == pbest_val[gbest_idx]:
            gbest_conv += 1
            
    gbest_val = pbest_val[gbest_idx]
    gbest_values = np.append(gbest_values, gbest_val)
    
    populasi = pbest_pop
    for j in range(v.shape[0]):
        for k in range(v.shape[1]):
            v[j,k] = (b_inersia * v[j,k]) + (c1 * random.random() * (pbest_val[j] - populasi[j,k])) + (c2 * random.random() * (gbest_val - populasi[j,k]))
            sig_v = 1 / (1 + pow(math.e, (-v[j,k])))
            if random.randint(0,1) < sig_v:
                populasi[j,k] = 1
            else:
                populasi[j,k] = 0
    
    print("Value Pbest : ", pbest_val)
    print("Nilai Gbest conv : ", gbest_conv)
    if gbest_conv == (len(pbest_val)):
        break
#SIMPAN TERM DARI GBEST
df = pd.DataFrame(gbest_values.T)
df.to_excel('Nilai Gbest Data Edit Pop1.xlsx', index='False')
gbest = pbest_pop[gbest_idx]
gbest_terms = []
gbest_W = []
for i in range(len(terms)):
    if gbest[i] == 1:
        gbest_terms.append(terms[i])
        gbest_W.append(W[i,:])
file = open('term gbest Data Edit Pop1.txt', 'w')
for term in gbest_terms:
    file.write('%s ' % term)
file.close()
gbest_W = np.array(gbest_W)
df = pd.DataFrame(gbest_W)
df.to_excel('Bobot Gbest Data Edit Pop1.xlsx', index='False')

exec_time = time.time() - starttime
seconds = exec_time % 60
minutes = exec_time // 60
hours = minutes // 60
minutes = minutes % 60
print("Total execution time akurasi Data Edit Pop1 : %d hours %d minutes %d seconds." % (hours, minutes, seconds))