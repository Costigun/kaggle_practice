#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[92]:


from sklearn.linear_model import LogisticRegression


# In[93]:


trainLabel = pd.read_csv("trainLabels.csv",header=None)
train = pd.read_csv("train.csv",header=None)
test = pd.read_csv('test.csv',header=None)


# In[94]:


train.head()


# In[95]:


train.info()


# In[96]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,train_test_split

X,y=train,np.ravel(trainLabel)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[97]:


model = LogisticRegression(max_iter=10000)


# In[98]:


train_accuracy = []
kfold = 10
val_accuracy = []
bestKnn = None
bestAcc = 0.0
neig = np.arange(1,25)

for i,k in enumerate(neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    
    knn.fit(X_train,y_train)
    
    train_accuracy.append(knn.score(X_train,y_train))
    
    val_accuracy.append(np.mean(cross_val_score(knn,X,y,cv=kfold)))
    if np.mean(cross_val_score(knn,X,y,cv=kfold)) > bestAcc:
        bestAcc = np.mean(cross_val_score(knn,X,y,cv=10))
        bestKnn = knn
    
print(bestAcc,bestKnn)


# Add feature scaling

# In[99]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer


# In[100]:


std = StandardScaler()

X_std = std.fit_transform(X)
mms = MinMaxScaler()
X_mms = mms.fit_transform(X)
norm = Normalizer()
X_norm = norm.fit_transform(X)


# In[102]:


neig = np.arange(1,30)
kfold = 10
val_accuracy = {'std':[],'mms':[],'norm':[]}
bestKnn = None
bestAcc = 0.0
bestScaling = None

for i,k in enumerate(neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    s1 = np.mean(cross_val_score(knn,X_std,y,cv=kfold))
    val_accuracy['std'].append(s1)
    s2 = np.mean(cross_val_score(knn,X_mms,y,cv=kfold))
    val_accuracy['mms'].append(s2)
    s3 = np.mean(cross_val_score(knn,X_norm,y,cv=kfold))
    val_accuracy['norm'].append(s3)
    if s1 > bestAcc:
        bestAcc = s1
        bestScaling = 'std'
        bestKnn = knn
    elif s2 > bestAcc:
        bestAcc = s2
        bestScaling = 'mms'
        bestKnn = knn
    elif s3 > bestAcc:
        bestAcc = s3
        bestScaling = 'norm'
        bestKnn = knn
plt.plot(neig,val_accuracy['std'],label='std')
plt.plot(neig,val_accuracy['mms'],label='mms')
plt.plot(neig,val_accuracy['norm'],label='norm')
plt.legend()


# In[103]:


print(bestKnn,bestAcc,bestScaling)


# In[29]:


model_knn = KNeighborsClassifier(n_neighbors=1000)


# In[30]:


model_knn.fit(X_train,y_train)


# In[32]:


X_train['Response'] = y_train


# In[40]:


import seaborn as sns


# In[43]:


plt.figure(figsize=(10,10))
sns.heatmap(X_train.corr(),cmap='viridis')


# In[53]:


best_cols = X_train.corr()[X_train.corr()['Response'].sort_values(ascending=False) > 0].index


# In[59]:


best_cols = best_cols.drop('Response')


# In[61]:


X_train = X_train[best_cols]


# In[62]:


model = LogisticRegression(max_iter=1000000)


# In[63]:


model.fit(X_train,y_train)


# In[65]:


X_test = X_test[best_cols]


# In[66]:


accuracy_score(model.predict(X_test),y_test)


# In[67]:


from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


# In[69]:


model_linear = LinearSVC()
model_linear.fit(X_train,y_train)
accuracy_score(model.predict(X_test),y_test)


# In[70]:


model_linear = SGDClassifier()
model_linear.fit(X_train,y_train)
accuracy_score(model.predict(X_test),y_test)


# In[ ]:




