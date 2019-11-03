#!/usr/bin/env python
# coding: utf-8

# In[13]:


import json
import matplotlib.pylab as plt
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import LinearSVC
#sparse
#Perceptron 0.8313 AveragedPerceptron 0.9433
#svm 0.9433
#winnow_acc_params [0.9196, 0.929, 0.9335, 0.9117, 0.5863]
#avg_winnow_acc_params [0.9433, 0.9374, 0.9282, 0.7544, 0.5463]
# adagrad_acc_params [0.9433, 0.9046, 0.7374, 0.5068, 0.5015]
#avg_adagrad_acc_params [0.9433, 0.8422, 0.7288, 0.5015, 0.5015]


#dense
#Perceptron 0.9194 AveragedPerceptron 0.9401
#svm 0.9401
#winnow_acc_params [0.9038, 0.9368, 0.9341, 0.9351, 0.8405]
#avg_winnow_acc_params [0.9401, 0.9401, 0.9386, 0.8953, 0.7049]
# adagrad_acc_params [0.9401, 0.9401, 0.6134, 0.4961, 0.4961]
#avg_adagrad_acc_params [0.9401, 0.8865, 0.6028, 0.4961, 0.4961]
# In[14]:

'''
def calculate_f1(y_gold, y_model):
    """
    Computes the F1 of the model predictions using the
    gold labels. Each of y_gold and y_model are lists with
    labels 1 or -1. The function should return the F1
    score as a number between 0 and 1.
    """
    y_gold=np.array(y_gold)
    y_model=np.array(y_model)
    TF=len(np.where(y_gold==1 and y_model==1)[0])
    TN=len(np.where(y_gold==-1 and y_model==-1)[0])
    FN=len(np.where(y_gold==1 and y_model==-1)[0])
    FP=len(np.where(y_gold==-1 and y_model==1)[0])
    precision=TP/len(TP+FP)
    recall=TP/len(TP+FN)
    F1=2*precision*recall/(precision+recall)
    return F1
'''
def calculate_f1(y_gold, y_model):
    """
    Computes the F1 of the model predictions using the
    gold labels. Each of y_gold and y_model are lists with
    labels 1 or -1. The function should return the F1
    score as a number between 0 and 1.
    """
    f1 =0.0
    precision= 0.0
    recall= 0.0
    
    TT=0
    FT=0
    FF=0
    TF=0
    for i in range(len(y_model)):
        if y_model[i] ==1:

            if y_model[i] != y_gold[i]:
                FT+=1
            else:
                TT+=1
        else: 
            if y_model[i] != y_gold[i]:
                FF+=1
            else: 
                TF+=1
    precision= TT/(FT+TT)
    recall = TT/(TT+FF)
    f1= 2*(precision * recall)/(precision+recall)
    return f1
#calculate_f1(y_pred, y)
# In[15]:


class Classifier(object):
    """
    The Classifier class is the base class for all of the Perceptron-based
    algorithms. Your class should override the "process_example" and
    "predict_single" functions. Further, the averaged models should
    override the "finalize" method, where the final parameter values
    should be calculated. You should not need to edit this class any further.
    """
    def train(self, X, y):
        iterations = 10
        for iteration in range(iterations):
            for x_i, y_i in zip(X, y):
                self.process_example(x_i, y_i)
        self.finalize()

    def process_example(self, x, y):
        """
        Makes a predicting using the current parameter values for
        the features x and potentially updates the parameters based
        on the gradient. "x" is a dictionary which maps from the feature
        name to the feature value and y is either 1 or -1.
        """
        
        raise NotImplementedError
        

    def finalize(self):
        """Calculates the final parameter values for the averaged models."""
        pass

    def predict(self, X):
        """
        Predicts labels for all of the input examples. You should not need
        to override this method.
        """
        y = []
        for x in X:
            y.append(self.predict_single(x))
        return y

    def predict_single(self, x):
        """
        Predicts a label, 1 or -1, for the input example. "x" is a dictionary
        which maps from the feature name to the feature value.
        """
        
        raise NotImplementedError


# In[16]:


class Perceptron(Classifier):
    def __init__(self, features):
        """
        Initializes the parameters for the Perceptron model. "features"
        is a list of all of the features of the model where each is
        represented by a string.
        """
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.eta = 1
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0

    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] += self.eta * y * value
            self.theta += self.eta * y

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1


# For the rest of the Perceptron-based algorithms, you will have to implement the corresponding class like we have done for "Perceptron".
# Use the "Perceptron" class as a guide for how to implement the functions.
'''
classifier = Perceptron(features)
classifier.train(X_train, y_train)
y_pred = classifier.predict(X_test)
perceptron_acc = accuracy_score(y_test, y_pred)
print('Perceptron', perceptron_acc)
'''
# In[17]:
'''
class AveragedPerceptron(Classifier):
    def __init__(self, features):
        self.eta = 1
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.w_acc = {feature: 0.0 for feature in features}
        self.theta_acc = 0
        self.mk=0
        self.M=0
        # You will need to add data members here
        
    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] += self.eta * y * value
            
            for feature in self.w:
                self.w_acc[feature]+=self.w[feature]*self.mk
            self.theta += self.eta * y
            self.theta_acc +=self.theta *self.mk
            self.mk=1
        else:
            self.mk+=1
            self.M+=1

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
        
    def finalize(self):
        #raise NotImplementedError
        for feature, value in self.w_acc.items():
                self.w[feature] =value/self.M
        self.theta=self.theta_acc/self.M

'''



class AveragedPerceptron(Classifier):
    def __init__(self, features):
        self.eta = 1
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.w_acc = {feature: 0.0 for feature in features}
        self.theta_acc = 0
       
        self.M=0
        # You will need to add data members here
        
    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] += self.eta * y * value
                self.w_acc[feature]+=self.eta * y * value*self.M
            self.theta += self.eta * y
            self.theta_acc +=self.eta * y*self.M
            
        else:
            
            self.M+=1

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
        
    def finalize(self):
        #raise NotImplementedError
        for feature, value in self.w_acc.items():
                self.w[feature] =self.w[feature]- value/self.M
        self.theta=self.theta-self.theta_acc/self.M
        


classifier = AveragedPerceptron(features)
classifier.train(X_dev, y_dev)
y_pred = classifier.predict(X_test)
avg_perceptron_acc = accuracy_score(y_test, y_pred)
print('AveragedPerceptron', avg_perceptron_acc)

# In[18]:


class Winnow(Classifier):
    def __init__(self, alpha, features):
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.alpha = alpha
        self.w = {feature: 1.0 for feature in features}
        self.theta = -len(features)
        
        
    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] *= self.alpha**( y * value)
           

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
'''
classifier = Winnow(1.1,features)
classifier.train(X_train, y_train)
y_pred = classifier.predict(X_test)
winnow_acc = accuracy_score(y_test, y_pred)
print('Winnow', winnow_acc)
'''
# In[19]:


class AveragedWinnow(Classifier):
    def __init__(self, alpha, features):
        self.alpha = alpha
        self.w = {feature: 1.0 for feature in features}
        self.theta = -len(features)
        self.w_acc = {feature: 0.0 for feature in features}
        self.M=0
        # You will need to add data members here
        
    def process_example(self, x, y):
        #raise NotImplementedError
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w_acc[feature]+=self.w[feature]*(self.alpha**( y * value)-1)*self.M
                self.w[feature] *= self.alpha**( y * value)
                
            
        else:
            
            self.M+=1

    def predict_single(self, x):
        #raise NotImplementedError
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
        
    def finalize(self):
        for feature, value in self.w_acc.items():
            self.w[feature] = self.w[feature]-value/self.M
        
        #raise NotImplementedError
'''
classifier = AveragedWinnow(1.1,features)
classifier.train(X_train, y_train)
y_pred = classifier.predict(X_test)
avg_winnow_acc = accuracy_score(y_test, y_pred)
print('AveragedWinnow', avg_winnow_acc)
'''
# In[20]:


class AdaGrad(Classifier):
    def __init__(self, eta, features):
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.eta = eta
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.G = {feature: 1e-5 for feature in features}  # 1e-5 prevents divide by 0 problems
        self.H = 0
        
    def process_example(self, x, y):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if y*score<=1:
            for feature, value in x.items():
                self.G[feature] += value**2
                self.w[feature] += self.eta * y * value/np.sqrt(self.G[feature])
            self.H+=1
            self.theta += self.eta * y/np.sqrt(self.H)

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1


'''
classifier = AdaGrad(1.5,features)
classifier.train(X_train, y_train)
y_pred = classifier.predict(X_test)
adagrad_acc = accuracy_score(y_test, y_pred)
print('AdaGrad', adagrad_acc)
'''
# In[21]:


class AveragedAdaGrad(Classifier):
    def __init__(self, eta, features):
        self.eta = eta
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.G = {feature: 1e-5 for feature in features}
        self.H = 0
        self.w_acc = {feature: 0.0 for feature in features}
        self.M=0
        self.theta_acc=0
        # You will need to add data members here
        
    def process_example(self, x, y):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if y*score<=1:
            for feature, value in x.items():
                self.G[feature] += value**2
                self.w[feature] += self.eta * y * value/np.sqrt(self.G[feature])
                self.w_acc[feature]+=self.eta * y * value/np.sqrt(self.G[feature])*self.M
            self.H+=1
            self.theta += self.eta * y/np.sqrt(self.H)
            self.theta_acc+=self.eta * y/np.sqrt(self.H)*self.M
        else:
            self.M+=1

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
        
    def finalize(self):
        #raise NotImplementedError
        for feature, value in self.w_acc.items():
                self.w[feature] =self.w[feature]-value/self.M
        self.theta=self.theta-self.theta_acc/self.M

classifier = AveragedAdaGrad(1.5,features)
classifier.train(X_dev, y_dev)
y_pred = classifier.predict(X_test)
avg_adagrad_acc = accuracy_score(y_test, y_pred)
print('AveragedAdaGrad', avg_adagrad_acc)

# In[22]:


def plot_learning_curves(perceptron_accs,
                         winnow_accs,
                         adagrad_accs,
                         avg_perceptron_accs,
                         avg_winnow_accs,
                         avg_adagrad_accs,
                         svm_accs):
    """
    This function will plot the learning curve for the 7 different models.
    Pass the accuracies as lists of length 11 where each item corresponds
    to a point on the learning curve.
    """
    assert len(perceptron_accs) == 11
    assert len(winnow_accs) == 11
    assert len(adagrad_accs) == 11
    assert len(avg_perceptron_accs) == 11
    assert len(avg_winnow_accs) == 11
    assert len(avg_adagrad_accs) == 11
    assert len(svm_accs) == 11

    x = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
    plt.figure()
    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
    ax.plot(x, perceptron_accs, label='perceptron')
    ax2.plot(x, perceptron_accs, label='perceptron')
    ax.plot(x, winnow_accs, label='winnow')
    ax2.plot(x, winnow_accs, label='winnow')
    ax.plot(x, adagrad_accs, label='adagrad')
    ax2.plot(x, adagrad_accs, label='adagrad')
    ax.plot(x, avg_perceptron_accs, label='avg-perceptron')
    ax2.plot(x, avg_perceptron_accs, label='avg-perceptron')
    ax.plot(x, avg_winnow_accs, label='avg-winnow')
    ax2.plot(x, avg_winnow_accs, label='avg-winnow')
    ax.plot(x, avg_adagrad_accs, label='avg-adagrad')
    ax2.plot(x, avg_adagrad_accs, label='avg-adagrad')
    ax.plot(x, svm_accs, label='svm')
    ax2.plot(x, svm_accs, label='svm')
    ax.set_xlim(0, 5500)
    ax2.set_xlim(49500, 50000)
    ax2.set_xticks([50000])
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    ax2.legend()


# In[23]:


def load_synthetic_data(directory_path):
    """
    Loads a synthetic dataset from the dataset root (e.g. "synthetic/sparse").
    You should not need to edit this method.
    """
    def load_jsonl(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def load_txt(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(int(line.strip()))
        return data

    def convert_to_sparse(X):
        sparse = []
        for x in X:
            data = {}
            for i, value in enumerate(x):
                if value != 0:
                    data[str(i)] = value
            sparse.append(data)
        return sparse

    X_train = load_jsonl(directory_path + '/train.X')
    X_dev = load_jsonl(directory_path + '/dev.X')
    X_test = load_jsonl(directory_path + '/test.X')

    num_features = len(X_train[0])
    features = [str(i) for i in range(num_features)]

    X_train = convert_to_sparse(X_train)
    X_dev = convert_to_sparse(X_dev)
    X_test = convert_to_sparse(X_test)

    y_train = load_txt(directory_path + '/train.y')
    y_dev = load_txt(directory_path + '/dev.y')
    y_test = load_txt(directory_path +  '/test.y')

    return X_train, y_train, X_dev, y_dev, X_test, y_test, features

X_train, y_train, X_dev, y_dev, X_test, y_test, features         = load_synthetic_data('C:/Users/yabinghu/Desktop/hw2-materials/synthetic/sparse')
# In[24]:


def run_synthetic_experiment(data_path):
    """
    Runs the synthetic experiment on either the sparse or dense data
    depending on the data path (e.g. "data/sparse" or "data/dense").
    
    We have provided how to train the Perceptron on the training and
    test on the testing data (the last part of the experiment). You need
    to implement the hyperparameter sweep, the learning curves, and
    predicting on the test dataset for the other models.
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test, features         = load_synthetic_data(data_path)
    
    
    '''
    # TODO: Train all 7 models on the training data and test on the test data
    # TODO: Hyperparameter sweeps
    classifier = Perceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    perceptron_acc = accuracy_score(y_dev, y_pred)
    print('Perceptron', perceptron_acc)
    
    classifier = AveragedPerceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    avg_perceptron_acc = accuracy_score(y_dev, y_pred)
    print('AveragedPerceptron', avg_perceptron_acc)
    
    winnow_acc_params=[]
    avg_winnow_acc_params=[]
    for alpha in [1.1,1.01,1.005,1.0005,1.0001]:
    
        classifier = Winnow(alpha,features)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_dev)
        winnow_acc = accuracy_score(y_dev, y_pred)
        winnow_acc_params.append(winnow_acc)
        print('Winnow', winnow_acc)
    
    
        classifier = AveragedWinnow(alpha,features)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_dev)
        avg_winnow_acc = accuracy_score(y_dev, y_pred)
        avg_winnow_acc_params.append(avg_winnow_acc)
        print('AveragedWinnow', avg_winnow_acc)
        
    
    adagrad_acc_params=[]
    avg_adagrad_acc_params=[]
    for eta in [1.5,0.25,0.03,0.005,0.001]:
        classifier = AdaGrad(eta,features)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_dev)
        adagrad_acc = accuracy_score(y_dev, y_pred)
        adagrad_acc_params.append(adagrad_acc)
        print('AdaGrad', adagrad_acc)
        
        classifier = AveragedAdaGrad(eta,features)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_dev)
        avg_adagrad_acc = accuracy_score(y_dev, y_pred)
        avg_adagrad_acc_params.append(avg_adagrad_acc)
        print('AveragedAdaGrad', avg_adagrad_acc)
        
        
       
    v = DictVectorizer()
    X = v.fit_transform(X_train)
    classifier =LinearSVC(loss='hinge')
    classifier.fit(X, y_train)
    X_test_new= v.fit_transform(X_dev)
    y_pred=classifier.predict(X_test_new)
    svm_acc=accuracy_score(y_dev, y_pred)
    print('svm', svm_acc)
    return perceptron_acc,avg_perceptron_acc,winnow_acc_params,avg_winnow_acc_params,adagrad_acc_params,avg_adagrad_acc_params,svm_acc
    '''
    
    # TODO: Placeholder data for the learning curves. You should write
    # the logic to downsample the dataset to the number of desired training
    # instances (e.g. 500, 1000), then train all of the models on the
    # sampled dataset. Compute the accuracy and add the accuraices to
    # the corresponding list.
    perceptron_accs = [0.1] * 11
    winnow_accs = [0.2] * 11
    adagrad_accs = [0.3] * 11
    avg_perceptron_accs = [0.4] * 11
    avg_winnow_accs = [0.5] * 11
    avg_adagrad_accs = [0.6] * 11
    svm_accs = [0.7] * 11
    
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train=X_train[indices]
    y_train=y_train[indices]
    '''
    X_dev=np.array(X_dev)
    y_dev=np.array(y_dev)
    indices = np.arange(X_dev.shape[0])
    np.random.shuffle(indices)
    X_dev=X_dev[indices]
    y_dev=y_dev[indices]
    '''
    
    nums=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
    for i in range(len(nums)):
        
        classifier = Perceptron(features)
        classifier.train(X_train[:nums[i]], y_train[:nums[i]])
        y_pred = classifier.predict(X_dev)
        perceptron_accs[i] = accuracy_score(y_dev, y_pred)
        
        
        classifier = AveragedPerceptron(features)
        classifier.train(X_train[:nums[i]], y_train[:nums[i]])
        y_pred = classifier.predict(X_dev)
        avg_perceptron_accs[i] = accuracy_score(y_dev, y_pred)
        
        alpha=1.005
        classifier = Winnow(alpha,features)
        classifier.train(X_train[:nums[i]], y_train[:nums[i]])
        y_pred = classifier.predict(X_dev)
        winnow_accs[i] = accuracy_score(y_dev, y_pred)
        
        
        alpha=1.1
        classifier = AveragedWinnow(alpha,features)
        classifier.train(X_train[:nums[i]], y_train[:nums[i]])
        y_pred = classifier.predict(X_dev)
        avg_winnow_accs[i]  = accuracy_score(y_dev, y_pred)
        
        eta=1.5
        classifier = AdaGrad(eta,features)
        classifier.train(X_train[:nums[i]], y_train[:nums[i]])
        y_pred = classifier.predict(X_dev)
        adagrad_accs[i] = accuracy_score(y_dev, y_pred)
        
        
        classifier = AveragedAdaGrad(eta,features)
        classifier.train(X_train[:nums[i]], y_train[:nums[i]])
        y_pred = classifier.predict(X_dev)
        avg_adagrad_accs[i] = accuracy_score(y_dev, y_pred)
        
        v = DictVectorizer()
        X = v.fit_transform(X_train[:nums[i]])
        classifier =LinearSVC(loss='hinge')
        classifier.fit(X, y_train[:nums[i]])
        X_test_new= v.fit_transform(X_dev)
        y_pred=classifier.predict(X_test_new)
        svm_accs[i]=accuracy_score(y_dev, y_pred)
        
    return perceptron_accs,winnow_accs,adagrad_accs,avg_perceptron_accs,avg_winnow_accs,avg_adagrad_accs,svm_accs
    plot_learning_curves(perceptron_accs, winnow_accs, adagrad_accs, avg_perceptron_accs, avg_winnow_accs, avg_adagrad_accs, svm_accs)
#perceptron_accs,winnow_accs,adagrad_accs,avg_perceptron_accs,avg_winnow_accs,avg_adagrad_accs,svm_accs=run_synthetic_experiment('C:/Users/yabinghu/Desktop/hw2-materials/synthetic/sparse')
perceptron_accs,winnow_accs,adagrad_accs,avg_perceptron_accs,avg_winnow_accs,avg_adagrad_accs,svm_accs=run_synthetic_experiment('C:/Users/yabinghu/Desktop/hw2-materials/synthetic/dense')
#perceptron_acc,avg_perceptron_acc,winnow_acc_params,avg_winnow_acc_params,adagrad_acc_params,avg_adagrad_acc_params,svm_acc=run_synthetic_experiment('C:/Users/yabinghu/Desktop/hw2-materials/synthetic/sparse')
#perceptron_acc,avg_perceptron_acc,winnow_acc_params,avg_winnow_acc_params,adagrad_acc_params,avg_adagrad_acc_params,svm_acc=run_synthetic_experiment('C:/Users/yabinghu/Desktop/hw2-materials/synthetic/dense')
# In[25]:


def load_ner_data(path):
    """
    Loads the NER data from a path (e.g. "ner/conll/train"). You should
    not need to edit this method.
    """
    # List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path + '/' + filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data
data=load_ner_data('ner/conll/train')

# In[26]:

'''
def extract_ner_features_train(train):
    """
    Extracts feature dictionaries and labels from the data in "train"
    Additionally creates a list of all of the features which were created.
    We have implemented the w-1 and w+1 features for you to show you how
    to create them.
    
    TODO: You should add your additional featurization code here.
    """
    y = []
    X = []
    features = set()
    for sentence in train:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        for i in range(1, len(padded) - 1):
            y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feats = [feat1, feat2]
            features.update(feats)
            feats = {feature: 1 for feature in feats}
            X.append(feats)
    return features, X, y
'''

def extract_ner_features_train(train):
    """
    Extracts feature dictionaries and labels from the data in "train"
    Additionally creates a list of all of the features which were created.
    We have implemented the w-1 and w+1 features for you to show you how
    to create them.
    
    TODO: You should add your additional featurization code here.
    """
    y = []
    X = []
    features = set()
    for sentence in train:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3, len(padded) - 3):
            y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-3=' + str(padded[i - 3][0])
            feat6 = 'w+3=' + str(padded[i + 3][0])
            feat7 = 'w-1=' + str(padded[i - 1][0])+'&'+'w-2=' + str(padded[i - 2][0])
            feat8 = 'w+1=' + str(padded[i + 1][0])+'&'+'w+2=' + str(padded[i + 2][0])
            feat9 = 'w-1=' + str(padded[i - 1][0])+'&'+'w+1=' + str(padded[i + 1][0])
            feats = [feat1, feat2,feat3,feat4,feat5, feat6,feat7, feat8,feat9]
            features.update(feats)
            feats = {feature: 1 for feature in feats}
            X.append(feats)
    return features, X, y
features, X, y=extract_ner_features_train(data)
# In[27]:


def extract_features_dev_or_test(data, features):
    """
    Extracts feature dictionaries and labels from "data". The only
    features which should be computed are those in "features". You
    should add your additional featurization code here.
    
    TODO: You should add your additional featurization code here.
    """
    y = []
    X = []
    for sentence in data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3, len(padded) - 3):
            y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-3=' + str(padded[i - 3][0])
            feat6 = 'w+3=' + str(padded[i + 3][0])
            feat7 = 'w-1=' + str(padded[i - 1][0])+'&'+'w-2=' + str(padded[i - 2][0])
            feat8 = 'w+1=' + str(padded[i + 1][0])+'&'+'w+2=' + str(padded[i + 2][0])
            feat9 = 'w-1=' + str(padded[i - 1][0])+'&'+'w+1=' + str(padded[i + 1][0])
            feats = [feat1, feat2,feat3,feat4,feat5, feat6,feat7, feat8,feat9]
            feats = {feature: 1 for feature in feats if feature in features}
            X.append(feats)
    return X, y

def extract_features_dev_or_test(data, features):
    """
    Extracts feature dictionaries and labels from "data". The only
    features which should be computed are those in "features". You
    should add your additional featurization code here.
    
    TODO: You should add your additional featurization code here.
    """
    y = []
    X = []
    '''
    for sentence in data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        for i in range(1, len(padded) - 1):
            y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feats = [feat1, feat2]
            feats = {feature: 1 for feature in feats if feature in features}
            X.append(feats)
    return X, y
    '''

    for sentence in data:
            padded = sentence[:]
            padded.insert(0, ('SSS', None))
            padded.insert(0, ('SSS', None))
            padded.insert(0, ('SSS', None))
            padded.append(('EEE', None))
            padded.append(('EEE', None))
            padded.append(('EEE', None))
            for i in range(3, len(padded) - 3):
                y.append(1 if padded[i][1] == 'I' else -1)
                feat1 = 'w-1=' + str(padded[i - 1][0])
                feat2 = 'w+1=' + str(padded[i + 1][0])
                feat3 ='w-2='+str(padded[i-2][0])
                feat4 ='w+2='+str(padded[i+2][0])
                feat5 ='w-3='+str(padded[i-3][0])
                feat6 ='w+3='+str(padded[i+3][0])
                feat7= feat1+'&'+ feat3
                feat8= feat2+'&'+ feat4
                feat9= feat1+'&'+ feat2
                
                feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            
                feats = {feature: 1 for feature in feats if feature in features}
                X.append(feats)
    return X, y
#data = load_ner_data('ner/conll/test')
#X, y=extract_features_dev_or_test(data,features)
# In[28]:


def run_ner_experiment(data_path):
    """
    Runs the NER experiment using the path to the ner data
    (e.g. "ner" from the released resources). We have implemented
    the standard Perceptron below. You should do the same for
    the averaged version and the SVM.
    
    The SVM requires transforming the features into a different
    format. See the end of this function for how to do that.
"""
    train = load_ner_data(data_path + '/conll/train')
    conll_test = load_ner_data(data_path + '/conll/test')
    enron_test = load_ner_data(data_path + '/enron/test')
    
    features, X_train, y_train = extract_ner_features_train(train)
    X_conll_test, y_conll_test = extract_features_dev_or_test(conll_test, features)
    X_enron_test, y_enron_test = extract_features_dev_or_test(enron_test, features)
             
    # You should do this for the Averaged Perceptron and SVM
    classifier = AveragedPerceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_conll_test)
    conll_f1 = calculate_f1(y_conll_test, y_pred)
    y_pred = classifier.predict(X_enron_test)
    enron_f1 = calculate_f1(y_enron_test, y_pred)
    print('Averaged Perceptron')
    print('  CoNLL', conll_f1)
    print('  Enron', enron_f1)
    
    # This is how you convert from the way we represent features in the
    # Perceptron code to how you need to represent features for the SVM.
    # You can then train with (X_train_dict, y_train) and test with
    # (X_conll_test_dict, y_conll_test) and (X_enron_test_dict, y_enron_test)
    vectorizer = DictVectorizer()
    X_train_dict = vectorizer.fit_transform(X_train)
    X_conll_test_dict = vectorizer.transform(X_conll_test)
    X_enron_test_dict = vectorizer.transform(X_enron_test)
    from sklearn.svm import LinearSVC
    classifier =LinearSVC(loss='hinge')
    classifier.fit(X_train_dict, y_train)
    y_pred = classifier.predict(X_conll_test_dict)
    conll_f1 = calculate_f1(y_conll_test, y_pred)
    y_pred = classifier.predict(X_enron_test_dict)
    enron_f1 = calculate_f1(y_enron_test, y_pred)
    print('SVM')
    print('  CoNLL', conll_f1)
    print('  Enron', enron_f1)
            
    run_ner_experiment('C:/Users/yabinghu/Desktop/hw2-materials/ner')    
''' 
v = DictVectorizer()
X_train_dict = v.fit_transform(X_train)
classifier =LinearSVC(loss='hinge')
classifier.fit(X_train_dict, y_train)
X_test_new= v.fit_transform(X_conll_test)
y_pred=classifier.predict(X_test_new)
svm_acc=accuracy_score(y_test, y_pred)



vectorizer = DictVectorizer()
X_train_dict = vectorizer.fit_transform(X_train)
X_conll_test_dict = vectorizer.transform(X_conll_test)
X_enron_test_dict = vectorizer.transform(X_enron_test)
from sklearn.svm import LinearSVC
classifier =LinearSVC(loss='hinge')
classifier.fit(X_train_dict, y_train)
y_pred = classifier.predict(X_conll_test_dict)
conll_f1 = calculate_f1(y_conll_test, y_pred)
y_pred = classifier.predict(X_enron_test)
enron_f1 = calculate_f1(X_enron_test_dict, y_pred)
print('SVM')
print('  CoNLL', conll_f1)
print('  Enron', enron_f1)
'''
# In[29]:


# Run the synthetic experiment on the sparse dataset. "synthetic/sparse"
# is the path to where the data is located.
run_synthetic_experiment('synthetic/sparse')


# In[30]:


# Run the synthetic experiment on the sparse dataset. "synthetic/dense"
# is the path to where the data is located.
run_synthetic_experiment('synthetic/dense')


# In[31]:


# Run the NER experiment. "ner" is the path to where the data is located.
run_ner_experiment('ner')


# In[ ]:
from sklearn.svm import LinearSVC
v = DictVectorizer()
X = v.fit_transform(X_train)
classifier =LinearSVC(loss='hinge')
classifier.fit(X, y_train)
X_test_new= v.fit_transform(X_test)
y_pred=classifier.predict(X_test_new)
svm_acc=accuracy_score(y_test, y_pred)



