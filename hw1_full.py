# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 07:20:54 2019

@author: yabinghu
"""
import numpy as np
from sklearn import tree
'''
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
clf.predict([[2, 2]])
'''


X_train=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/train-X.npy')
Y_train=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/train-y.npy')
X_test=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/test-X.npy')
Y_test=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/test-y.npy')
#%%
#2_i

from sklearn.linear_model import SGDClassifier
model_sgd = SGDClassifier(loss='log', max_iter=10000)
model_sgd = model_sgd.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score
Y_pred=model_sgd.predict(X_train)
accuracy_score(Y_train, Y_pred)

#%%
#2_ii
model_dt = tree.DecisionTreeClassifier(criterion='entropy')
model_dt = model_dt.fit(X_train,Y_train)
Y_pred=model_dt.predict(X_train)
accuracy_score(Y_train, Y_pred)

#%%
#2_iii
model_dsd = tree.DecisionTreeClassifier(criterion='entropy',max_depth=4)
model_dsd = model_dt.fit(X_train,Y_train)
Y_pred=model_dsd.predict(X_train)
accuracy_score(Y_train, Y_pred)
#%%
#2_iv
stumps=[]
X_prime_train=[]
X_prime_test=[]
for i in range(50):
    index=np.random.choice(X_train.shape[1], int(0.5*X_train.shape[1]), replace=False)
    X_train_sub=X_train[ : , index]
    model_dsf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=4)
    stumps.append(model_dsf.fit(X_train_sub,Y_train))
    X_prime_train.append(stumps[i].predict(X_train_sub))
    X_prime_test.append(stumps[i].predict(X_test[ : , index]))

X_prime_train=np.array(X_prime_train).T
X_prime_test=np.array(X_prime_test).T
model_sgd_new = SGDClassifier(loss='log', max_iter=10000)
model_sgd_new = model_sgd_new.fit(X_prime_train,Y_train)



Y_pred_test=model_sgd_new.predict(X_prime_test)
accuracy_score(Y_pred_test, Y_test)

Y_pred=model_sgd_new.predict(X_prime_train)
accuracy_score(Y_pred, Y_train)

#%%
cv_held_X0=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-X.0.npy')
cv_held_X1=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-X.1.npy')
cv_held_X2=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-X.2.npy')
cv_held_X3=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-X.3.npy')
cv_held_X4=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-X.4.npy')
cv_held_Y0=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-y.0.npy')
cv_held_Y1=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-y.1.npy')
cv_held_Y2=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-y.2.npy')
cv_held_Y3=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-y.3.npy')
cv_held_Y4=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-heldout-y.4.npy')

cv_train_X0=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-X.0.npy')
cv_train_X1=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-X.1.npy')
cv_train_X2=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-X.2.npy')
cv_train_X3=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-X.3.npy')
cv_train_X4=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-X.4.npy')
cv_train_Y0=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-y.0.npy')
cv_train_Y1=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-y.1.npy')
cv_train_Y2=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-y.2.npy')
cv_train_Y3=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-y.3.npy')
cv_train_Y4=np.load('C:/Users/yabinghu/Desktop/hw1-materials/madelon/cv-train-y.4.npy')

#%%
#cross validation
CV_train_X=[cv_train_X0,cv_train_X1,cv_train_X2,cv_train_X3,cv_train_X4]
CV_train_Y=[cv_train_Y0,cv_train_Y1,cv_train_Y2,cv_train_Y3,cv_train_Y4]
CV_held_X=[cv_held_X0,cv_held_X1,cv_held_X2,cv_held_X3,cv_held_X4]
CV_held_Y=[cv_held_Y0,cv_held_Y1,cv_held_Y2,cv_held_Y3,cv_held_Y4]
acc_train_sgd=[]
acc_held_sgd=[]
acc_train_dt=[]
acc_held_dt=[]
acc_train_dsd=[]
acc_held_dsd=[]
acc_train_sgd_new=[]
acc_held_sgd_new=[]



for i in range(5):
    model_sgd = SGDClassifier(loss='log', max_iter=10000)
    model_sgd = model_sgd.fit(CV_train_X[i],CV_train_Y[i])
    Y_pred=model_sgd.predict(CV_train_X[i])
    cur_acc=len(np.where(CV_train_Y[i]==Y_pred)[0])/Y_pred.shape[0]
    acc_train_sgd.append(cur_acc)
    Y_pred=model_sgd.predict(CV_held_X[i])
    cur_held=len(np.where(CV_held_Y[i]==Y_pred)[0])/Y_pred.shape[0]
    acc_held_sgd.append(cur_held)
    
    model_dt = tree.DecisionTreeClassifier(criterion='entropy')
    model_dt = model_dt.fit(CV_train_X[i],CV_train_Y[i])
    Y_pred=model_dt.predict(CV_train_X[i])
    cur_acc=len(np.where(CV_train_Y[i]==Y_pred)[0])/Y_pred.shape[0]
    acc_train_dt.append(cur_acc)
    Y_pred=model_dt.predict(CV_held_X[i])
    cur_held=len(np.where(CV_held_Y[i]==Y_pred)[0])/Y_pred.shape[0]
    acc_held_dt.append(cur_held)
    
    
    model_dsd = tree.DecisionTreeClassifier(criterion='entropy',max_depth=4)
    model_dsd = model_dsd.fit(CV_train_X[i],CV_train_Y[i])
    Y_pred=model_dsd.predict(CV_train_X[i])
    cur_acc=len(np.where(CV_train_Y[i]==Y_pred)[0])/Y_pred.shape[0]
    acc_train_dsd.append(cur_acc)
    Y_pred=model_dsd.predict(CV_held_X[i])
    cur_held=len(np.where(CV_held_Y[i]==Y_pred)[0])/Y_pred.shape[0]
    acc_held_dsd.append(cur_held)
    
     
    stumps=[]
    X_prime_train=[]
    X_prime_test=[]
    for k in range(50):
        index=np.random.choice(CV_train_X[i].shape[1], int(0.5*CV_train_X[i].shape[1]), replace=False)
        
        X_train_sub=CV_train_X[i][ : , index]
        model_dsf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=4)
        stumps.append(model_dsf.fit(X_train_sub,CV_train_Y[i]))
        X_prime_train.append(stumps[k].predict(X_train_sub))
        X_prime_test.append(stumps[k].predict(CV_held_X[i][ : , index]))
    
    X_prime_train=np.array(X_prime_train).T
    X_prime_test=np.array(X_prime_test).T
    model_sgd_new = SGDClassifier(loss='log', max_iter=10000)
    model_sgd_new = model_sgd_new.fit(X_prime_train,CV_train_Y[i])
    
    
    
    Y_pred_test=model_sgd_new.predict(X_prime_test)
    acc_held_sgd_new.append(accuracy_score(Y_pred_test,CV_held_Y[i]))
    
    Y_pred=model_sgd_new.predict(X_prime_train)
    acc_train_sgd_new.append(accuracy_score(Y_pred, CV_train_Y[i]))
    
sgd_train_acc=sum(acc_train_sgd)/5
sgd_train_std=np.std(np.array(sgd_train_acc))
sgd_heldout_acc=sum(acc_held_sgd)/5
sgd_heldout_std=np.std(np.array(acc_held_sgd))
Y_pred=model_sgd.predict(X_test)
sgd_test_acc=accuracy_score(Y_test, Y_pred)

dt_train_acc=sum(acc_train_dt)/5
dt_train_std=np.std(np.array(acc_train_dt))
dt_heldout_acc=sum(acc_held_dt)/5
dt_heldout_std=np.std(np.array(acc_held_dt))
Y_pred=model_dt.predict(X_test)
dt_test_acc=accuracy_score(Y_test, Y_pred)

dt4_train_acc=sum(acc_train_dsd)/5
dt4_train_std=np.std(np.array(acc_train_dsd))
dt4_heldout_acc=sum(acc_held_dsd)/5
dt4_heldout_std=np.std(np.array(acc_held_dsd))
Y_pred=model_dsd.predict(X_test)
dt4_test_acc=accuracy_score(Y_test, Y_pred)

stumps_train_acc=sum(acc_train_sgd_new)/5
stumps_train_std=np.std(np.array(acc_train_sgd_new))
stumps_heldout_acc=sum(acc_held_sgd_new)/5
stumps_heldout_std=np.std(np.array(acc_held_sgd_new))

stumps=[]
X_prime_train=[]
X_prime_test=[]
for i in range(50):
    index=np.random.choice(X_train.shape[1], int(0.5*X_train.shape[1]), replace=False)
    X_train_sub=X_train[ : , index]
    model_dsf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=4)
    stumps.append(model_dsf.fit(X_train_sub,Y_train))
    X_prime_train.append(stumps[i].predict(X_train_sub))
    X_prime_test.append(stumps[i].predict(X_test[ : , index]))

X_prime_train=np.array(X_prime_train).T
X_prime_test=np.array(X_prime_test).T
model_sgd_new = SGDClassifier(loss='log', max_iter=10000)
model_sgd_new = model_sgd_new.fit(X_prime_train,Y_train)

Y_pred_test=model_sgd_new.predict(X_prime_test)
stumps_test_acc=accuracy_score(Y_pred_test,Y_test)







#%%
import os
import matplotlib.pyplot as plt


def plot_results(sgd_train_acc, sgd_train_std, sgd_heldout_acc, sgd_heldout_std, sgd_test_acc,
                 dt_train_acc, dt_train_std, dt_heldout_acc, dt_heldout_std, dt_test_acc,
                 dt4_train_acc, dt4_train_std, dt4_heldout_acc, dt4_heldout_std, dt4_test_acc,
                 stumps_train_acc, stumps_train_std, stumps_heldout_acc, stumps_heldout_std, stumps_test_acc):
    """
    Plots the final results from problem 2. For each of the 4 classifiers, pass
    the training accuracy, training standard deviation, held-out accuracy, held-out
    standard deviation, and testing accuracy.

    Although it should not be necessary, feel free to edit this method.
    """
    train_x_pos = [0, 4, 8, 12]
    cv_x_pos = [1, 5, 9, 13]
    test_x_pos = [2, 6, 10, 14]
    ticks = cv_x_pos

    labels = ['sgd', 'dt', 'dt4', 'stumps (4 x 50)']

    train_accs = [sgd_train_acc, dt_train_acc, dt4_train_acc, stumps_train_acc]
    train_errors = [sgd_train_std, dt_train_std, dt4_train_std, stumps_train_std]

    cv_accs = [sgd_heldout_acc, dt_heldout_acc, dt4_heldout_acc, stumps_heldout_acc]
    cv_errors = [sgd_heldout_std, dt_heldout_std, dt4_heldout_std, stumps_heldout_std]

    test_accs = [sgd_test_acc, dt_test_acc, dt4_test_acc, stumps_test_acc]

    fig, ax = plt.subplots()
    ax.bar(train_x_pos, train_accs, yerr=train_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='train')
    ax.bar(cv_x_pos, cv_accs, yerr=cv_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='held-out')
    ax.bar(test_x_pos, test_accs, align='center', alpha=0.5, capsize=10, label='test')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_title('Models')
    ax.yaxis.grid(True)
    ax.legend()
    plt.tight_layout()    
    



plot_results(sgd_train_acc, sgd_train_std, sgd_heldout_acc,sgd_heldout_std, sgd_test_acc,dt_train_acc, dt_train_std, dt_heldout_acc, dt_heldout_std, dt_test_acc,dt4_train_acc, dt4_train_std, dt4_heldout_acc, dt4_heldout_std, dt4_test_acc,stumps_train_acc, stumps_train_std, stumps_heldout_acc, stumps_heldout_std, stumps_test_acc)

  
#%%
import numpy as np


# When you turn this function in to Gradescope, it is easiest to copy and paste this cell to a new python file called hw1.py
# and upload that file instead of the full Jupyter Notebook code (which will cause problems for Gradescope)
def compute_features(names):
    N=len(names)
    matrix=np.zeros([N,260])
    dict_={}
    k=0
    for char in 'abcdefghijklmnopqrstuvwxyz':
        dict_[char]=k
        k+=1
    for i in range(N):
        temp=names[i].split(' ')
        first_name=temp[0]
        last_name=temp[1]
        for k in range(5):
            if k<=len(first_name)-1:
                matrix[i][dict_[first_name[k]]+k*26]=1
            else:
                matrix[i][k*26:(k+1)*26]=np.zeros(26)
            if k<=len(last_name)-1:
                matrix[i][dict_[last_name[k]]+130+k*26]=1
            else:
                matrix[i][k*26+130:(k+1)*26+130]=np.zeros(26) 
    return matrix

Y_train=np.load('C:/Users/yabinghu/Desktop/hw1-materials/badges/train.labels.npy')
with open('C:/Users/yabinghu/Desktop/hw1-materials/badges/train.names.txt', 'r') as f:
    X_train = f.readlines()
X_train=[x.replace('\n','') for x in X_train ]

Y_test=np.load('C:/Users/yabinghu/Desktop/hw1-materials/badges/test.labels.npy')
with open('C:/Users/yabinghu/Desktop/hw1-materials/badges/test.names.txt', 'r') as f:
    X_test = f.readlines()
X_test=[x.replace('\n','') for x in X_test ]

X_train_new=compute_features(X_train)