import numpy as np
import matplotlib.pyplot as plt
import math
from PCA import PCA

data = np.genfromtxt('audioData.csv', delimiter=",")

#if encounter a missing value, use the mean of feature instead
fmeans = np.nanmean(data, axis=0)
indexNan = np.argwhere(np.isnan(data))
for i in indexNan:
    #print(i)
    data[i] = fmeans[i[1]]

def InitialCenters(k,data):
    #centers = np.zeros((k,data.shape[1]))
    centers = data[np.random.choice(data.shape[0], k, replace=False), :]
    return centers


#KMEANS
def assignLabels(data, centers):
    dist = np.zeros((data.shape[0],centers.shape[0]))
    distance = np.zeros((data.shape[0],1))
    labels = np.zeros((data.shape[0],1))
    for i in range(data.shape[0]):
            dist[i] = np.sqrt(np.sum(np.square(data[i]-centers), axis = 1))
            distance[i] = np.amin(dist[i])
    labels = np.argmin(dist, axis=1)
    return labels, distance

def updateCenter(labels, data, k):
    centers = np.zeros((k,data.shape[1]))
    for i in range(k):
        s = np.zeros((1,data.shape[1]))
        count = 0
        for j in range(labels.shape[0]):
            if labels[j] == i:
                s = np.add(s, data[j])
                count = count + 1
        centers[i] = s/count
    return centers

def updateK(data):
    obj = np.zeros((9,1))
    #obj1 = np.zeros((9,1))
    for k in range(2,11):
        #print(k)
        #print('#################################################################')
        centers = InitialCenters(k,data)
        #print(centers)
        assign = assignLabels(data, centers)
        labels = assign[0]
    #    obj1[k-2] = np.sum(np.square(assign[1]))
        #print(labels)
        while np.array_equal(updateCenter(labels, data, k), centers) == False:
            centers = updateCenter(labels, data, k)
            assign = assignLabels(data, centers)
            labels = assign[0]
            print('continue....')
        print('done')
        dismin = assign[1] 
        obj[k-2] = np.sum(np.square(dismin))
    print(obj)
    #print(obj1)
    return obj

def kmeansPlot(obj):
    plt.figure(1)
    plt.ylabel('objective function')
    plt.xlabel('K value')
    plt.plot(np.arange(2,11,1), obj, 'k')
    plt.show()

##without PCA
#obj = updateK(data)
#kmeansPlot(obj)
#
#with PCA
#data_pca = PCA(data)
#obj_pca = updateK(data_pca)
#kmeansPlot(obj_pca)
#GMM 


#initial p(u), assuming both are 0.5
Pu = np.array([0.5, 0.5])
E = np.zeros((128,2))

#do E-step
def estep(data, Pu):
    E1 = np.zeros((128,2))
    
    datacov = np.cov(data.T)
    covInverse = np.linalg.inv(datacov)
    covDet = np.linalg.det(datacov)
    Cond = np.zeros((128,2))
    Denominator = np.zeros((128,1))  
    for i in range(128):
        for j in range(2):
            s1 = np.matmul((data[i]-centers[j]).T,covInverse)
            s2 = 1/((2*math.pi)**(13/2)*math.sqrt(covDet))
            s3 = -0.5*np.matmul(s1, (data[i]-centers[j]))
            s4 = s2 * math.exp(s3)
            Cond[i][j] = s4
    for i in range(128):
        Denominator[i] = Cond[i][0] * Pu[0] + Cond[i][1] * Pu[1]
        for j in range(2):
            E1[i][j] = Cond[i][j]*Pu[j]/Denominator[i]
    return E1

#do M-step
def mstep(Pu, E, data, centers):
    for i in range(2):
        entity = np.multiply(data, E[:,i].reshape(128,1))
        Sum = np.sum(entity, axis=0)
        centers[i] = Sum/np.sum(E[:,i])
    Pu = np.sum(E, axis=0)/128
    return Pu, centers

#print(Pu)
def updateGMM(data, Pu, E, centers):
    count = 0
    print(Pu)
    while np.array_equal(estep(data, Pu), E) == False:
        E = estep(data, Pu)
        Mresult = mstep(Pu, E, data, centers)
        Pu = Mresult[0]
        centers = Mresult[1]
        print(centers)
    #    print('continue....')
        count = count + 1
        
    print('done')
    print(count)
    return E 

#plot
def plotGmm(data, E):
    plt.figure(2)
    for i in range(128):
        if E[i][0] > E[i][1]:
            plt.scatter(data[i][0], data[i][1] , c = 'r')
        else:
            plt.scatter(data[i][0], data[i][1] , c = 'b')


           
#without PCA
#initial means
#centers = InitialCenters(2,data)
#print(centers)
#E = updateGMM(data, Pu, E, centers)
#plotGmm(data, E)
#with PCA
data_p = PCA(data)
centers = InitialCenters(2,data_p)
E_p = updateGMM(data_p, Pu, E, centers)
plotGmm(data_p, E)
    
    
    
    
    
    
    