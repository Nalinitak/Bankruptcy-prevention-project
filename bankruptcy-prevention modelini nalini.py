#!/usr/bin/env python
# coding: utf-8

# #import Basic labrary

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[2]:


data=pd.read_csv("bankruptcy-prevention (1).csv",sep=";")
data


# # Basic EDA
# 

# In[3]:


data.info()


# In[4]:


data.columns


# In[5]:


data.isna()


# In[6]:


data.isna().sum()


# In[7]:


data.duplicated()


# In[8]:


data.duplicated().sum()


# In[9]:


data.describe(include='all')


# In[10]:


##Count of duplicated rows
data[data.duplicated()].shape


# In[11]:


data[data.duplicated()]


# In[12]:


data.sum()


# In[13]:


data.dtypes


# In[14]:


data.corr()


# In[15]:


sns.heatmap(data.corr(), annot=True)
plt.show()


# In[16]:


data.boxplot(column=['industrial_risk'])


# In[17]:


#Box plot
data.plot(kind='box')
plt.show()


# In[18]:


data.hist()


# In[19]:


data.plot(kind='kde')


# In[20]:


data.plot.bar()


# In[21]:


data[' class'].value_counts()


# In[ ]:





# In[22]:


data.industrial_risk.value_counts()


# In[23]:


data.industrial_risk.value_counts()


# In[24]:


import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[25]:


# Dividing our data into input and output variables 
X = data.iloc[:,0:5]
Y = data.iloc[:,6]


# In[26]:


X


# In[27]:


Y


# In[28]:


from sklearn.model_selection import train_test_split
train_x,test_x=train_test_split(X,test_size=0.3)
train_Y,test_Y=train_test_split(Y,test_size=0.3)


# In[29]:


train_Y,test_Y


# In[30]:


#Logistic regression and fit the model
classifier = LogisticRegression()
classifier.fit(X,Y)


# In[31]:


#Predict for X dataset
y_prob = classifier.predict_proba(X)
y_pred= classifier.predict(X)

pd.DataFrame(y_prob)


# In[32]:


y_pred


# In[33]:


y_pred_data= pd.DataFrame({'actual': Y,
                         'predicted': classifier.predict(X)})
y_pred_data


# In[34]:


y_pred_data
pd.crosstab(y_pred_data.actual,y_pred_data.predicted)


# In[35]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(Y,y_pred)
print (cm)
print(accuracy_score(Y,y_pred))


# In[36]:


import numpy as np
Accuracy= np.sum([cm[0,0],cm[1,1]])/np.sum(cm)
print(Accuracy)


# In[37]:


TN=cm[0,0]
TP=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]
sensitivity=TP/(TP+FN)
spec=TN/(TN+FP)#specificity
precision=TP/(TP+FP)#+ve precision
print(sensitivity,spec,precision)


# In[38]:


f=(2*sensitivity*precision)/(sensitivity+precision)
f


# In[39]:


cm_test=confusion_matrix(test_Y,classifier.predict(test_x))
print(cm_test)


# In[40]:


#Classification report
from sklearn.metrics import classification_report
print(classification_report(Y,y_pred))


# In[41]:





# In[42]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:





# In[51]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





