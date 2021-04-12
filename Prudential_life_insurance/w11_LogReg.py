#!/usr/bin/env python
# coding: utf-8

# Проверим качество предсказания через каппа-метрику и матрицу неточностей.
# 
# Данные:
# * https://video.ittensive.com/machine-learning/prudential/train.csv.gz
# 
# Соревнование: https://www.kaggle.com/c/prudential-life-insurance-assessment/
# 
# © ITtensive, 2020

# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


# In[19]:


data = pd.read_csv("https://video.ittensive.com/machine-learning/prudential/train.csv.gz")


# In[20]:


print(data.info())


# In[21]:


data["Product_Info_2_1"] = data["Product_Info_2"].str.slice(0, 1)
data["Product_Info_2_2"] = pd.to_numeric(data["Product_Info_2"].str.slice(1, 2))
data.drop(labels=["Product_Info_2"], axis=1, inplace=True)
print (data.info())


# In[22]:


def reduce_mem_usage (df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print ("Потребление памяти меньше на", round(start_mem - end_mem, 2), "Мб (минус)", round(100*(start_mem - end_mem) / start_mem, 1), "%)")
    return df


# In[23]:


data = reduce_mem_usage(data)


# In[24]:


for l in data['Product_Info_2_1'].unique():
    data['Product_Info_2_1' + l] = data['Product_Info_2_1'].isin([l]).astype('int8')
data.drop('Product_Info_2_1',axis=1,inplace=True)


# In[25]:


columns_groups = ['Insurance_History','Insured_Info','Medical_Keyword',
                  'Family_Hist','Medical_History','Product_Info']
columns = ['Wt','Ht','BMI','Ins_Age']

for cg in columns_groups:
    columns.extend(data.columns[data.columns.str.startswith(cg)])
print(columns)


# In[26]:


data.fillna(-1,inplace=True)


# In[27]:


scaler = preprocessing.StandardScaler()
scaler.fit(pd.DataFrame(data,columns=columns))


# In[28]:


data_train,data_test = train_test_split(data,test_size=0.2)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)


# In[29]:


data.isnull().sum().sort_values()


# In[30]:


def regression_model(df,columns):
    y = df['Response']
    x = scaler.transform(pd.DataFrame(df,columns=columns))
    model = LogisticRegression(max_iter=1000,class_weight='balanced',multi_class='multinomial')
    model.fit(x,y)
    return model
def logistic_regression(columns):
    x = scaler.transform(data_test[columns])
    model = regression_model(data_train,columns) 
    data_test['target'] = model.predict(x)
    return cohen_kappa_score(data_test['target'],data_test['Response'],weights='quadratic')


# In[31]:


print(logistic_regression(columns))


# In[32]:


print(confusion_matrix(data_test['target'],data_test['Response']))


# In[ ]:




