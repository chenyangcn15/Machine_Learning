# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:26:22 2018

@author: Chen
"""
import numpy as np

def PCA(x):
    m = np.mean(x, axis=0)
    c = x - m
    v = np.cov(c.T)
    values, vectors = np.linalg.eig(v)
    x_p = vectors.T.dot(c.T)
    return x_p.T[:,0:2]

#A = np.array([[1,2],[3,4],[5,6]])
#A_p = PCA(A)