# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 02:35:57 2018

@author: Chen
"""
import random
import nltk
from collections import Counter
import os
from os.path import join
import re
import math
import pandas as pd
#import numpy as np
nltk.download("stopwords")          # Download the stop words from nltk

 # English stopwords from nltk
stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['the','to','is','a','of','it','at','in','me','as']
stopwords.extend(newStopWords)
#print(stopwords) 

pattern = re.compile(r'([a-zA-Z]+)')
def words(text): return re.findall(pattern, text.lower())


#shuffle files
def splitdata(filepath, ratio, trainfeature, testfeatureinonefile):
    for root, dirs, files in os.walk(filepath):
        random.shuffle(files)
        num = int(len(files) * ratio)
        trainfile, testfile = files[num:],files[:num]
        trainraw = []
        testeachfile = []
        for file in trainfile:     
           # print(file)
            temp = words(open(os.path.join(filepath,file), encoding="utf8").read())
            #corpus = Counter(temp)
            for w in temp: 
                if(w not in stopwords and len(w)>3):
                    trainraw.append(w)
                    #print(len(trainraw))
        for file2 in testfile:      
           # print(file2+'**************************')
            temp2 = words(open(os.path.join(filepath,file2), encoding="utf8").read())
            #corpus2 = Counter(temp2)
            testraw = []
            for w2 in temp2: 
                if(w2 not in stopwords and len(w2)>3):
                    testraw.append(w2)
            testeachfile.append(testraw)
        return {'trainfeature':trainraw,'testfeatureinonefile':testeachfile}
    #print(len(trainfeature))
    #print(len(testfeature))
    
#cal condiprob
#feature is a matrix .its entry is (filename, word, occurance)
def condiprob(Pfeature,Nfeature, Tfeature,V,condipro):
    #print('here')
    #pcondipro = []
    #ncondipro = []
    pprior = []
    nprior = []
    for w in Tfeature:
        #print(w)
        if w in Pfeature: #condi prob for word is positive 
            #print('in postitive, check', Pfeature[w])
            #prior = math.log(Pfeature[w]+1)-math.log(len(Pfeature)+V)
            pprior.append([w,(Pfeature[w]+1)/(len(Pfeature)+V)])
        #prior = math.log(1)-math.log(len(Pfeature)+V)
        else:
            pprior.append([w,1/(len(Pfeature)+V)])
        #print(prior)
        #pcondipro.append(prior)
        if w in Nfeature: #condi prob for word is neg
            nprior.append([w,(Nfeature[w]+1)/(len(Nfeature)+V)])
        else:
            nprior.append([w,(1)/(len(Nfeature)+V)])
        #ncondipro.append(prior1)
    #for i in range(0,len(Tfeature)):
        #print(pprior[i],nprior[i])
    condipro.append(pprior)
    condipro.append(nprior)
    print(len(pprior),len(nprior))
    return condipro

trainPfeature = []
testPfeature = []
trainNf=[]
testNf=[]
trainTraw = []
trainTf=[]
ratio = 0.7 
#ratio=0.5           
pfp=r'C:\Users\Chen\Downloads\CSE575 statistic machine learning\HW1\movie review data(1)\movie review data\pos'
r1=splitdata(pfp,ratio,trainPfeature,testPfeature)
trainPfeature = Counter(r1['trainfeature'])
#print(len(trainPfeature))
testPfeature = r1['testfeatureinonefile']

#print(len(trainPfeature))
nfp=r'C:\Users\Chen\Downloads\CSE575 statistic machine learning\HW1\movie review data(1)\movie review data\neg'
r2=splitdata(nfp,ratio,trainNf,testNf)
trainNf = Counter(r2['trainfeature'])
#print(len(trainNf))
#trainTf = Counter(pd.concat([r1['trainfeature'],r2['trainfeature']]))
testNf = r2['testfeatureinonefile']

for w in r2['trainfeature'] or r1['trainfeature']:
    trainTraw.append(w)
trainTf = Counter(trainTraw)

V = len(trainPfeature)+len(trainNf)
for w in trainPfeature:
    if w in trainNf:
        V = V-1
#print(V)   
cp = []
cp=condiprob(trainPfeature, trainNf, trainTf, V, cp)
np=[]
pp=[]
pp=cp[0]
np=cp[1]

#print(cp[0][0])

#test
p_pos=1
p_neg=1
p_pos1=1
p_neg1=1
error=0
for i in range(0,len(testNf)):
    matrix = testNf[i]
    for w in matrix:
        for k,v in pp:
            if w==k:
                p_pos=v*p_pos 
        for k,v in np:
            if w==k:
                p_neg=v*p_neg
    if p_pos>p_neg:
        error=error+1
for i in range(0,len(testPfeature)):
    matrix1 = testPfeature[i]
    for w in matrix1:
        for k,v in pp:
            if w==k:
                p_pos1=v*p_pos1 
        for k,v in np:
            if w==k:
                p_neg1=v*p_neg1
    if p_pos1<p_neg1:
        error=error+1
print(error)
accurancy=1-error/(len(testNf)+len(testPfeature))
