import numpy as np
#from scipy.special import expit
from load_dataset import read, show

[train_lab,train_img]=read()
[test_lab,test_img]=read("testing")

#reshape to nx784 matrix
train_img=train_img.reshape(train_img.shape[0],(train_img.shape[1]* train_img.shape[2])).astype(int)
test_img=test_img.reshape(test_img.shape[0],(test_img.shape[1]* test_img.shape[2])).astype(int)

#for test
#train_tl=train_lab[:10000]
#train_ti=train_img[:10000]
#test_tl=test_lab[:5000]
#test_ti=test_img[:5000]
#print(train_tl.shape,train_ti.shape,test_tl.shape,test_ti.shape)

#in logistic regression, all we need is find this w (ignore b)
w= np.zeros((10,784),dtype=float)

#final predict function
def sigmoid(w,X):
    return (1 / (1 + np.exp(-np.matmul(X, w)))).astype(float)

#find w
#first step: relabel to 0,1. such that for 5, y[4][i] is a 0-1 label array.
y = np.zeros((10, len(train_lab)),dtype=int)
for k in range(10):
    for i in range(len(train_lab)):
        if train_lab[i] == k:
            y[k][i] = 1
        else:
            y[k][i] = 0

#update w
step = 0.001
p=np.zeros((10,train_img.shape[0]),dtype=float)
difference = np.zeros((10,train_img.shape[0]))
J=np.zeros((10,784),dtype=float)
# the loop is update. change range(i) with 1,3,5,7,9,12,15,...
for iteration in range(100):
    for i in range(10):
        #print("executing...")
        p[i]=sigmoid(w[i],train_img)
        difference[i] = y[i]-p[i]
        #print(p[i].shape)
        J[i]= np.matmul(difference[i],train_img)
        #J[i]=np.dot((p[i]-y[i]),train_ti)
        #print(J[i].shape)
        w[i]=w[i]+step*J[i]

def result_accuracy(pre, test_label):
    error=0
    print("calculating accuracy...")
    for i in range(len(pre)):
        if ((int(pre[i])-test_label[i]) != 0):
            error=error+1
    accuracy=1-error/(len(pre))
    return accuracy

#predict       
fp=np.zeros((10,10000),dtype=float)
for i in range(10):        
    fp[i]= np.matmul(test_img,w[i])
    #print(np.matmul(test_img,w[i]))
result=np.argmax(fp, axis=0)
accuracy=result_accuracy(result,test_lab)
print(accuracy)

