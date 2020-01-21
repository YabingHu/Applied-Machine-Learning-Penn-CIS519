#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from sklearn.metrics import accuracy_score
import numpy as np
# In[10]:
def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    # TODO
    dict={}
    res={'<unk>'}
    for doc in D:
        for word in doc:
            if word not in dict:
                dict[word]=1
            else:
                dict[word]+=1
            if dict[word]>1:
                res.add(word)
    return res
        

# In[1]:


class BBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the binary bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        #raise NotImplementedError
        feature_dictionary={}
        for token in doc:
            if token in vocab:
                feature_dictionary[token]=1
            else:
                if '<unk>' in vocab:
                    feature_dictionary['<unk>']=1
        return feature_dictionary
                
        


# In[2]:


class CBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the count bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        feature_dictionary={}
        for token in doc:
            if token in vocab:
                if token in feature_dictionary:
                    feature_dictionary[token]+=1
                else:
                    feature_dictionary[token]=1 
            else:
                if '<unk>' in feature_dictionary:
                    feature_dictionary['<unk>']+=1
                else:
                    feature_dictionary['<unk>']=1
                    
        return feature_dictionary
        


# In[3]:


def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    # TODO
    # raise NotImplementedError
    import numpy as np
    IDFs={}
    d=len(D)
    for doc in D:
        doc=list(set(doc))
        flag=0
        for token in doc:
            if token in vocab:
                if token in IDFs:
                    
                    IDFs[token]+=1
                else:
                    IDFs[token]=1
            else:
                if '<unk>' in IDFs:
                    if flag==0:
                        IDFs['<unk>']+=1
                        flag=1
                    
                else:
                    IDFs['<unk>']=1   
                    flag=1     
    for token in IDFs:
        IDFs[token]=np.log(d/IDFs[token])
    return IDFs
            
    
    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        # raise NotImplementedError
        feature_dictionary={}
        for token in doc:
            if token in vocab:
                if token in feature_dictionary:
                    feature_dictionary[token]+=1
                else:
                    feature_dictionary[token]=1 
            else:
                if '<unk>' in feature_dictionary:
                    feature_dictionary['<unk>']+=1
                else:
                    feature_dictionary['<unk>']=1
        for ele in  feature_dictionary:
             feature_dictionary[ele]= feature_dictionary[ele]*self.idf[ele]
        return feature_dictionary
        

        
        
        


# In[2]:


# You should not need to edit this cell
def load_dataset(file_path):
    D = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            D.append(instance['document'])
            y.append(instance['label'])
    return D, y

def convert_to_features(D, featurizer, vocab):
    X = []
    for doc in D:
        X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
    return X


# In[ ]:


def train_naive_bayes(X, y, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    vocab is the set of vocabulary tokens.
    7
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the innner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    # TODO
    #raise NotImplementedError
    # TODO
    #raise NotImplementedError
    
    
    n=len(y)
    p_y={}
    cnt_y=[ [l, y.count(l)] for l in set(y)]
    for ele in cnt_y:
        p_y[ele[0]]=ele[1]/n
    V=len(vocab)
    p_v_y={}
    for sub_y in list(set(y)):
        p_v_y[sub_y]={}
    m=len(list(set(y)))
    y=np.array(y)
    X=np.array(X)
    for i in range(m):
        x_d_y_d= X[np.where(y==i)[0]]
        f_d_v={}
        for v in vocab:
            f_d_v[v]=0
        f_d_w=0
        for x in x_d_y_d:
            f_d_w+=sum(x.values())
            for ele in x:
                f_d_v[ele]+=x[ele]
        for ele in f_d_v:
            p_v_y[i][ele]=(k+f_d_v[ele])/(f_d_w+k*V)
    return p_y,p_v_y     


# In[ ]:

def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    import numpy as np
    pred =[] # list of integer labels
    p_y_d =[] # list of floats
    for doc in D: #[[a,d,f],[a,c]...]
        temp ={}
        p_d =0
        for i in p_y:
            temp[i]=0
            for word in doc:
                if word in p_v_y[i]:
                    temp[i] += np.log(p_v_y[i][word])
                else:
                    temp[i] += np.log(p_v_y[i]['<unk>']) 
            temp[i]+=np.log(p_y[i])
            p_d+= np.exp(temp[i])
        max_value = max(temp.values())
        for inx in temp:
            if temp[inx] == max_value:
                max_inx = inx   
        p_y_d.append(np.exp(max_value)/p_d)
        pred.append(int(max_inx)) #  0 or 1

    
    return pred, p_y_d 


# In[ ]:
def train_semi_supervised(X_sup, y_sup, D_unsup, X_unsup, D_valid, y_valid, k, vocab, mode):
    """
    Trains the Naive Bayes classifier using the semi-supervised algorithm.
    
    X_sup: A list of the featurized supervised documents.
    y_sup: A list of the corresponding supervised labels.
    D_unsup: The unsupervised documents.
    X_unsup: The unsupervised document representations.
    D_valid: The validation documents.
    y_valid: The validation labels.
    k: The smoothing parameter for Naive Bayes.
    vocab: The vocabulary as a set of tokens.
    mode: either "threshold" or "top-k", depending on which selection
        algorithm should be used.
    
    Returns the final p_y and p_v_y (see `train_naive_bayes`) after the
    algorithm terminates.    
    """
    # TODO
    #raise NotImplementedError
    
    p_y,p_v_y =train_naive_bayes(X_sup, y_sup, k, vocab)
    U_y,p_y_d=predict_naive_bayes(D_valid, p_y, p_v_y)
    acc=accuracy_score(y_valid, U_y)
    print('the initial accuracy is'+ ' ',acc)
    
    if mode =='threshold':
        threshold=0.98
        while True:
            #S_x_new=[]
            #idx=[]
            p_y,p_v_y =train_naive_bayes(X_sup, y_sup, k, vocab)
            U_y,p_y_d=predict_naive_bayes(D_unsup, p_y, p_v_y)
            idx=list(np.where(np.array(p_y_d)>=threshold)[0])
            idx_left=list(np.where(np.array(p_y_d)<threshold)[0])
            if idx:
            	S_y_new=list(np.array(U_y)[idx])
            	S_x_new=list(np.array(X_unsup)[idx])
            #if not S_x_new:
            if not idx:
            	return p_y,p_v_y
            X_sup.extend(S_x_new)
            y_sup.extend(S_y_new)
            D_unsup=list(np.array(D_unsup)[idx_left])
            
            
    if mode =='top_k':
        K=10000
        while True:
            S_x_new=[]
            idx=[]
            p_y,p_v_y =train_naive_bayes(X_sup, y_sup, k, vocab)
            U_y,p_y_d=predict_naive_bayes(D_unsup, p_y, p_v_y)
            confidence=sorted(range(len(p_y_d)), key=lambda x: p_y_d[x],reverse=True)
            if len(confidence)>=K:
                idx=confidence[:K]
                idx_left=confidence[K:]
            if idx:
                S_x_new=list(np.array(X_unsup)[np.array(idx)])
                S_y_new=list(np.array(U_y)[np.array(idx)])
            if not S_x_new:
            	return p_y,p_v_y
            X_sup.extend(S_x_new)
            y_sup.extend(S_y_new)
            D_unsup=list(np.array(D_unsup)[idx_left])


# In[11]:


# Variables that are named D_* are lists of documents where each
# document is a list of tokens. y_* is a list of integer class labels.
# X_* is a list of the feature dictionaries for each document.
D_train, y_train = load_dataset('C:/Users/yabinghu/Desktop/hw4-materials/data/train.jsonl')
D_valid, y_valid = load_dataset('C:/Users/yabinghu/Desktop/hw4-materials/data/valid.jsonl')
D_test, y_test = load_dataset('C:/Users/yabinghu/Desktop/hw4-materials/data/test.jsonl')

vocab = get_vocabulary(D_train)


# In[ ]:


# Compute the features, for example, using the BBowFeaturizer.
# You actually only need to conver the training instances to their
# feature-based representations.
# 
# This is just starter code for the experiment. You need to fill in
# the rest.

featurizer = BBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)
bb_acc=[]
for k in [0.001,0.01,0.1,1.0,10]:
    p_y,p_v_y=train_naive_bayes(X_train, y_train, k, vocab)
    pred,p_y_d=predict_naive_bayes(D_valid, p_y, p_v_y)
    acc=accuracy_score(y_valid, pred)
    print(acc)
    bb_acc.append(acc)
    
#%%
featurizer = BBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)
p_y,p_v_y=train_naive_bayes(X_train, y_train, 0.1, vocab)
pred,p_y_d=predict_naive_bayes(D_test, p_y, p_v_y)
bb_test_acc=accuracy_score(y_test, pred)
print(bb_test_acc)
#%%
featurizer = CBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)
cb_acc=[]
for k in [0.001,0.01,0.1,1.0,10]:
    p_y,p_v_y=train_naive_bayes(X_train, y_train, k, vocab)
    pred,p_y_d=predict_naive_bayes(D_valid, p_y, p_v_y)
    acc=accuracy_score(y_valid, pred)
    print(acc)
    cb_acc.append(acc)
    
#%%
featurizer = CBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)
p_y,p_v_y=train_naive_bayes(X_train, y_train, 0.1, vocab)
pred,p_y_d=predict_naive_bayes(D_test, p_y, p_v_y)
cb_test_acc=accuracy_score(y_test, pred)
print(cb_test_acc)

#%%

featurizer = TFIDFFeaturizer(compute_idf(D_train,vocab))
X_train = convert_to_features(D_train, featurizer, vocab)
tf_acc=[]
for k in [0.001,0.01,0.1,1.0,10]:
    p_y,p_v_y=train_naive_bayes(X_train, y_train, k, vocab)
    pred,p_y_d=predict_naive_bayes(D_valid, p_y, p_v_y)
    acc=accuracy_score(y_valid, pred)
    print(acc)
    tf_acc.append(acc)

#%%
featurizer = TFIDFFeaturizer(compute_idf(D_train,vocab))
X_train = convert_to_features(D_train, featurizer, vocab)
p_y,p_v_y=train_naive_bayes(X_train, y_train, 1.0, vocab)
pred,p_y_d=predict_naive_bayes(D_test, p_y, p_v_y)
tf_test_acc=accuracy_score(y_test, pred)
print(tf_test_acc)
#%%


 

#%%
p=[50,500,5000]
acc_k=[]
acc_t=[]
for partition in p:
    
    
    indices = np.arange(len(D_train))
    np.random.shuffle(indices)
    
    D_train = list(np.array(D_train)[indices])
    y_train= list(np.array(y_train)[indices])
    
    featurizer = CBoWFeaturizer()
    X_train = convert_to_features(D_train, featurizer, vocab)
    
    X_sup=X_train[:partition]
    y_sup=y_train[:partition]
    D_unsup=D_train[partition:]
    X_unsup=X_train[partition:]
    
    
    p_y,p_v_y=train_semi_supervised(X_sup, y_sup, D_unsup, X_unsup, D_valid, y_valid, 0.1, vocab, 'top_k')
    pred,p_y_d=predict_naive_bayes(D_valid, p_y, p_v_y)
    acc=accuracy_score(y_valid, pred)
    acc_k.append(acc)
    print(acc)
    
    p_y,p_v_y=train_semi_supervised(X_sup, y_sup, D_unsup, X_unsup, D_valid, y_valid, 0.1, vocab, 'threshold')
    pred,p_y_d=predict_naive_bayes(D_valid, p_y, p_v_y)
    acc=accuracy_score(y_valid, pred)
    acc_t.append(acc)
    print(acc)

#%%
C=[1,6,15,20,15,6,1]
p_totoal=0
for x in range(7):
    p_totoal+=C[x]*(8+x)/((6+x)+8)
p=p_totoal/64
