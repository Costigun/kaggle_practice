#!/usr/bin/env python
# coding: utf-8

# Загрузите данные и разделите выборку на обучающую/проверочную в соотношении 80/20.
# 
# Примените метод ближайших соседей (kNN) для классификации скоринга, используйте k=100. Используйте биометрические данные, все столбцы Insurance_History, Family_Hist, Medical_History и InsurеdInfo. Заполните отсутствующие значения -1.
# 
# Проведите предсказание и проверьте качество через каппа-метрику.
# 
# Данные:
# Данные:
# * https://video.ittensive.com/machine-learning/prudential/train.csv.gz
# 
# Соревнование: https://www.kaggle.com/c/prudential-life-insurance-assessment/
# 
# © ITtensive, 2020
# Вопросы к этому заданию
# Какое значение каппа метрики с точностью до сотых ? Например, 0,00-0,01

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


data = pd.read_csv("https://video.ittensive.com/machine-learning/prudential/train.csv.gz")


# In[5]:


data.fillna(-1,inplace=True)


# In[38]:


columns = ['Wt','Ht','Ins_Age','BMI','Response']
for i in data.columns[data.columns.str.startswith('Insurance_History')]:
    columns.append(i)
for i in data.columns[data.columns.str.startswith('Family_Hist')]:
    columns.append(i)
for i in data.columns[data.columns.str.startswith('Medical_History')]:
    columns.append(i)
for i in data.columns[data.columns.str.startswith('InsuredInfo')]:
    columns.append(i)    


# In[39]:


model = KNeighborsClassifier(n_neighbors=100)


# In[43]:


data = pd.DataFrame(data,columns=columns)


# In[44]:


data_train,data_test = train_test_split(data,test_size=0.2)


# In[45]:


x = data_train.drop('Response',axis=1)
y = data_train['Response']


# In[46]:


model.fit(x,y)


# In[47]:


x_test = data_test.drop('Response',axis=1)


# In[50]:


y_pred = model.predict(x_test)


# In[52]:


cohen_kappa_score(y_pred,data_test['Response'],weights='quadratic')


# In[ ]:




