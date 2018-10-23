# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:10:00 2018

@author: Chen
"""
import matplotlib.pyplot as plt


plt.figure(1)
plt.ylabel('accuracy')
plt.xlabel('K value')
plt.plot([1,3,5,10,30,50,70,80,90,100], [0.9691,0.9717,0.9693,0.9683,0.9603,0.9544,0.9494,0.9473,0.9458,0.9444], 'bo',[1,3,5,10,30,50,70,80,90,100], [0.9691,0.9717,0.9693,0.9683,0.9603,0.9544,0.9494,0.9473,0.9458,0.9444], 'k')
plt.show()

plt.figure(2)
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.plot([1,3,5,7,8,9,12,15,20,30,50,70,80,90,100],[0.6804,0.6804,0.7861,0.7217,0.687,0.6267,0.6637,0.7664,0.7411,0.7598,0.7692,0.8631,0.8084,0.8321,0.8625], 'k',[1,3,5,7,8,9,12,15,20,30,50,70,80,90,100],[0.6804,0.6804,0.7861,0.7217,0.687,0.6267,0.6637,0.7664,0.7411,0.7598,0.7692,0.8631,0.8084,0.8321,0.8625], 'ro')
plt.show()