# -*- coding: utf-8 -*-
import nltk
from collections import Counter
import os
from os.path import join
import re
#import numpy as np
nltk.download("stopwords")          # Download the stop words from nltk

 # English stopwords from nltk
stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['the','to','is','a','of','it','at','in','me','as']
stopwords.extend(newStopWords)
#print(stopwords) 
    
pattern = re.compile(r'([a-zA-Z]+)')
def words(text): return re.findall(pattern, text.lower())
def load_data(filepath,features):
    for root, dirs, files in os.walk(filepath):
        for file in files:           
            temp = words(open(os.path.join(filepath,file), encoding="utf8").read())
            corpus = Counter(temp)
            for (w,c) in corpus.items(): 
                if(w not in stopwords and len(w)>3):
                    features.append([file,w,c])
               
pf = [] # data matrix for positive class               
pfp=r'C:\Users\Chen\Downloads\CSE575 statistic machine learning\HW1\movie review data(1)\movie review data\pos'
load_data(pfp,pf)
nf = [] # data matrix for negative class               
nfp=r'C:\Users\Chen\Downloads\CSE575 statistic machine learning\HW1\movie review data(1)\movie review data\neg'
load_data(nfp,nf)
with open('matrix.txt', 'w') as outfile:
    for i in range(0,len(pf)):
        outfile.write("%s %s %d\n" % (pf[i][0],pf[i][1],pf[i][2]))
    for j in range(0,len(nf)):
        outfile.write("%s %s %d\n" % (nf[j][0],nf[j][1],nf[j][2]))
outfile.close()


            

