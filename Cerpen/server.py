from flask import Flask, Response, request, abort, send_from_directory, render_template, send_file
import pandas as pd
import numpy as np
import random
import math
import time
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)
_author__ = 'May Ressureccion'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()
kategori = ["Anak", "Fantasi", "Selain Anak & Fantasi"]

label_training = np.array([])
for i in range(525):
    if i < 175:
        label_training = np.append(label_training, 0)
    elif i >= 175 and i < 350:
        label_training = np.append(label_training, 1)
    else:
        label_training = np.append(label_training, 2)
gnb = GaussianNB()
W = np.array(pd.read_excel('bobot awal.xlsx'))
gnb.fit(W, label_training)
gnb_pso = GaussianNB()
W_PSO = np.array(pd.read_excel('bobot pso1.xlsx')).T
gnb_pso.fit(W_PSO, label_training)

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

temp_f = open('terms.txt', 'r')
terms_awal = []
for f in temp_f:
    f = f.strip()
    terms_awal.append(f)
terms_awal = np.array(terms_awal[0].split(' '))

temp_g = open('terms-pso1.txt', 'r')
terms_awal_pso = []
for f in temp_g:
    f = f.strip()
    terms_awal_pso.append(f)
terms_awal_pso = np.array(terms_awal_pso[0].split(' '))

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

def pembobotan(datas, pso=False):
    terms = terms_awal
    if pso == True:
        terms = terms_awal_pso
    tf = np.zeros((terms.size, len(datas)))
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
    return W_testing

@app.route('/')
def index():
    return render_template(
        'index.html'
    )

@app.route('/NB')
def NaiveBayes():
    nama_cerpen = ""
    kategori_cerpen = ""
    proc_time = ""
    return render_template(
        'NB.html',
        nama_cerpen = nama_cerpen,
        kategori_cerpen = kategori_cerpen,
        proc_time = proc_time
    )

@app.route('/NB-PSO')
def NaiveBayesPSO():
    nama_cerpen = ""
    kategori_cerpen = ""
    proc_time = ""
    return render_template(
        'NB-PSO.html',
        nama_cerpen = nama_cerpen,
        kategori_cerpen = kategori_cerpen,
        proc_time = proc_time
    )

@app.route('/koleksi')
def koleksi():
    koleksi = []
    cerpen_path = ["koleksi-cerpen/Anak/", "koleksi-cerpen/Fantasi/", "koleksi-cerpen/Tidak Diketahui/"]
    for path in cerpen_path:
        koleksi.append(np.array([os.path.join(path,fname) for fname in os.listdir(path) if fname.endswith('.txt')]))
    len1 = koleksi[0].size
    len2 = koleksi[1].size
    len3 = koleksi[2].size
    nama_koleksi = [np.array([]), np.array([]), np.array([])]
    for i in range(len1):
        nama_koleksi[0] = np.append(nama_koleksi[0], koleksi[0][i].split("/")[-1].split(".")[0])
    for i in range(len2):
        nama_koleksi[1] = np.append(nama_koleksi[1], koleksi[1][i].split("/")[-1].split(".")[0])
    for i in range(len3):
        nama_koleksi[2] = np.append(nama_koleksi[2], koleksi[2][i].split("/")[-1].split(".")[0])
    print(koleksi)
    return render_template(
        'Koleksi.html',
        nama_koleksi = nama_koleksi,
        koleksi = koleksi,
        len1 = len1,
        len2 = len2,
        len3 = len3
    )
    
@app.route('/return-files/<path:path>')
def return_files(path):
	try:
		return send_from_directory(("/").join(path.split("/")[:-1]), path.split("/")[-1], as_attachment=True)
	except Exception as e:
		return str(e)

@app.route('/NB/NB-hasil', methods=['POST'])
def NBhasil():
    start_time = time.time()
    cerpen_path = ["koleksi-cerpen/Anak/", "koleksi-cerpen/Fantasi/", "koleksi-cerpen/Tidak Diketahui/"]
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
    W_testing = pembobotan(datas)
    result = int(gnb.predict(W_testing)[0])
    kategori_cerpen = kategori[result]
    class_dir = os.path.join(APP_ROOT, cerpen_path[result])
    class_dest = '/'.join([class_dir, filename])
    file.save(class_dest)
    proc_time = time.time() - start_time
    proc_time = str(proc_time) + ' seconds.'
    return render_template(
        'NB.html',
        nama_cerpen = filename,
        kategori_cerpen = kategori_cerpen,
        proc_time = proc_time
    )

@app.route('/NB-PSO/NB-PSO-hasil', methods=['POST'])
def NBPSO_hasil():
    start_time = time.time()
    cerpen_path = ["koleksi-cerpen/Anak/", "koleksi-cerpen/Fantasi/", "koleksi-cerpen/Tidak Diketahui/"]
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
    W_testing = pembobotan(datas, pso=True)
    result = int(gnb_pso.predict(W_testing)[0])
    kategori_cerpen = kategori[result]
    class_dir = os.path.join(APP_ROOT, cerpen_path[result])
    class_dest = '/'.join([class_dir, filename])
    file.save(class_dest)
    proc_time = time.time() - start_time
    proc_time = str(proc_time) + ' seconds.'
    return render_template(
        'NB-PSO.html',
        nama_cerpen = filename,
        kategori_cerpen = kategori_cerpen,
        proc_time = proc_time
    )

if __name__ == '__main__':
    app.run(port=5000, debug=True)