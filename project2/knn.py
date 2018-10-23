import numpy as np
from load_dataset import read, show


[train_lab,train_img]=read()
[test_lab,test_img]=read("testing")

#for test
#train_lab=train_lab[:1000]
#train_img=train_img[:1000]
#test_lab=test_lab[:500]
#test_img=test_img[:500]


#print(train_tl.shape,train_ti.shape,test_tl.shape,test_ti.shape)

#reshape to nx784 matrix
train_img=train_img.reshape(train_img.shape[0],(train_img.shape[1]* train_img.shape[2])).astype(int)
test_img=test_img.reshape(test_img.shape[0],(test_img.shape[1]* test_img.shape[2])).astype(int)


def getNearestKDistance(train_data, test_data,k):
    nearKDI = np.zeros((test_data.shape[0],k),dtype=int)
    for i in range(test_data.shape[0]):
        all_dist = np.sqrt(np.sum(np.square(test_data[i]-train_data),axis=1)).tolist()
        nearDisIndex = np.argsort(all_dist)
        nearKDI[i] = nearDisIndex[:k]
    return nearKDI
    #return all_dist, shortDisIndex
  
##choose the label that have maximum number in circle
def predict(train_label, nearKDI,k):
    print("the best choice is ...")
    pre = []
    for i in range(nearKDI.shape[0]):
        #every new test initial labelCount to empty dict
        labelCount = {}
        for j in range(k):
            label = train_label[nearKDI[i][j]]
            labelCount[str(label)] = labelCount.get(str(label),0)+1
            #sort
        sortedlabelcount = sorted(labelCount, key=labelCount.__getitem__, reverse=True)
        pre.append(sortedlabelcount[0][0])
    return pre

      

def result_accuracy(pre, test_label):
    error=0
    print("calculating accuracy...")
    for i in range(len(pre)):
        if ((int(pre[i])-test_label[i]) != 0):
            error=error+1
    accuracy=1-error/(len(pre))
    return accuracy

#let k = 1,3,5,10,30,50,70,80,90,100
nkd=getNearestKDistance(train_img,test_img,50) 
pre=predict(train_lab, nkd,50)
accuracy=result_accuracy(pre,test_lab)
print(accuracy)

