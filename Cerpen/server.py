from flask import Flask, Response, request, abort, send_from_directory, render_template
import pandas as pd
import numpy as np
import random
import math
import time
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)
_author__ = 'May Ressureccion'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()
kategori = ["Anak", "Fantasi", "Tidak Diketahui"]

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

temp_f = open('terms.txt', 'r')
terms_awal = []
for f in temp_f:
    f = f.strip()
    terms_awal.append(f)
terms_awal = np.array(terms_awal[0].split(' '))

def read_data(kalimat):
    datas = []
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
    datas.append(data)
    return datas

@app.route('/')
def index():
    return render_template(
        'index.html'
    )

@app.route('/NB')
def NaiveBayes():
    return render_template(
        'NB.html'
    )

@app.route('/NB-PSO')
def NaiveBayesPSO():
    nama_cerpen = ""
    kategori_cerpen = ""
    return render_template(
        'NB-PSO.html',
        nama_cerpen = nama_cerpen,
        kategori_cerpen = kategori_cerpen
    )

@app.route('/koleksi')
def koleksi():
    return render_template(
        'Koleksi.html'
    )

@app.route('/NB/NB-hasil', methods=['POST'])
def NBhasil():
    start_time = time.time()
    target = os.path.join(APP_ROOT, 'uploads/')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files.getlist('file')[0]
    print(file)
    filename = file.filename
    destination = '/'.join([target, filename])
    file.save(destination)
    contents = open('.\\uploads\\'+filename, 'r')
    kalimat = []
    for c in contents:
        c = c.strip()
        kalimat.append(c)
    datas = read_data(kalimat)
    W = np.array(pd.read_excel('bobot awal.xlsx'))
    data = datas[0]
    P = [np.zeros(terms_awal.size)]
    for i in range(P[0].size):
        if np.argwhere(data == terms_awal[i]).size > 0:
            P[0][i] = 1
    total_used_W_a = []
    total_used_W_f = []
    total_used_W_td = []
    for j in range(len(P)):
        tmp_a = []
        tmp_f = []
        tmp_td = []
        for k in range(P[0].size):
            if P[j][k] == 1:
                tmp_a.append(sum(W[:175,k]))
                tmp_f.append(sum(W[175:350,k]))
                tmp_td.append(sum(W[350:,k]))
        total_used_W_a.append(tmp_a)
        total_used_W_f.append(tmp_f)
        total_used_W_td.append(tmp_td)
    all_W = [total_used_W_a[0], total_used_W_f[0], total_used_W_td[0]]
    result = naive_bayes(all_W, P[0])
    kategori_cerpen = kategori[np.argmax(result)]
    return render_template(
        'NB.html',
        nama_cerpen = filename,
        kategori_cerpen = kategori_cerpen
    )
    

if __name__ == '__main__':
    app.run(port=1010, debug=True)