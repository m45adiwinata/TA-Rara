# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 01:09:02 2019

@author: Grenceng
"""

import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

data_path = open('data path.txt', 'r')
path = []
for p in data_path:
    p = p.strip()
    path.append(p)
cerpens = []
for p in path:
    cerpens.append([os.path.join(p,fname) for fname in os.listdir(p) if fname.endswith('.txt')])

datas= []
for cerpen in cerpens:
    for cer in cerpen:
        contents = open(cer, 'r')
        kalimat = []
        for c in contents:
            c = c.strip()
            kalimat.append(c)
        data = []
        for k in kalimat:
            katastop = stopword.remove(k)
            katadasar = stemmer.stem(katastop)
            katastop = stopword.remove(katadasar)
            data.append(katastop.split(' '))
        datas.append(data)