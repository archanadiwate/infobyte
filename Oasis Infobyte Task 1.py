#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION task 1
#    Iris flower has three species; setosa, versicolor, and virginica, which differs according to their measurements. Now assume that you have the measurements of the iris flowers according to their species, and here your task is to train a machine learning model that can learn from the measurements of the iris species and classify them.

# In[4]:


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm


# In[5]:


data=pd.read_csv(r"E:\Iris.csv")
data


# # Exploring data

# In[6]:


data.shape


# In[7]:


data.size


# In[8]:


data.head()


# # Obtaining Description/Summary Dataframe

# In[9]:


data.describe()


# In[10]:


data.info()


# # Viewing unique categories/values

# In[11]:


unique=data.Species.unique()
print(len(unique))
print(unique)


# # Taking required numnerical data into another dataframe

# In[12]:


data_df=data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]  #choosing ftures having numerical data
data_df.sample(5) #viewing random 5 records from dataset


# # Pairplot

# In[13]:


sns.pairplot(data)


# In[14]:


data.df2=data.drop(['Id'],axis=1)
plt.figure(figsize=(10,10))
sns.pairplot(data.df2,hue='Species',size=3)

From the above pairplotit shows that features of iris-Setosa are distinguishable from features of other categories
# # Heat Map
#    Heat Map allows you to visualize how storngy/weakly or positively/negatively the feature are correlated with light to dark colour & value of correlation coefficients

# In[16]:


plt.figure(figsize=(8,8))
corre=data.df2.corr()
sns.heatmap(corre,annot=True) #to visualize the stronge/weak correlation exists & annot=True to show that correlation value


# In[17]:


sns.FacetCrid(data,true='Species').map(plt.scatter,'Sepal Length','Sepal Width').add_legend()
plt.show()


sns.FacetCrid(data,true='Species').map(plt.scatter,'Petal Length','Petal Width').add_legend()
plt.show()


# In[18]:


plt.figure(figsize=(8,8))
sns.boxplot(y='PetalLengthCm',x='Species',data=data.df2)


# In[19]:


plt.figure(figsize=(8,8))
sns.boxplot(y='PetalWidthCm',x='Species',data=data.df2)


# In[20]:


plt.figure(figsize=(8,8))
sns.boxplot(y='SepalLengthCm',x='Species',data=data.df2)


# In[21]:


plt.figure(figsize=(8,8))
sns.boxplot(y='SepalWidthCm',x='Species',data=data.df2)

From the above plot we see that outlier present
# # VIOLIN PLOT

# In[22]:


plt.figure(figsize=(8,8))
sns.violinplot(y='PetalLengthCm',x='Species',data=data.df2)
plt.show()

plt.figure(figsize=(8,8))
sns.violinplot(y='PetalWidthCm',x='Species',data=data.df2)
plt.show()

plt.figure(figsize=(8,8))
sns.violinplot(y='SepalLengthCm',x='Species',data=data.df2)
plt.show()

plt.figure(figsize=(8,8))
sns.violinplot(y='SepalWidthCm',x='Species',data=data.df2)
plt.show()


# # Splitting of data into train & test sets

# In[25]:


x= data.df2.iloc[:, :-1].values
y= data.df2.iloc[:, -1].values


# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0) #80-20 ratio


# In[27]:


[len(x_train),len(x_test),len(y_train),len(y_test)]


# # Using K Means of Clusterring

# In[28]:


from sklearn.cluster import KMeans
css=[]
X=data.iloc[: , [0,1,2,3]].values
for i in range(1,11):
    Kmeans= KMeans(n_clusters = i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    Kmeans.fit(X)
    css.append(Kmeans.inertia_)
    
plt.plot(range(1,11), css)
plt.show()


# In[29]:


from sklearn.neighbors import KNeighborsClassifier

k=3
kclassifier=KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
kclassifier


# In[30]:


y_pred=kclassifier.predict(x_test)
compare_df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
compare_df.sample(10)


# In[31]:


cnt=0
index=[]
for i,j in zip(compare_df['Actual'],compare_df['Predicted']):
    if(i!=j):
        cnt+=1
        print(cnt)


# # Evaluate K Classifier Model

# In[32]:


from sklearn.metrics import accuracy_score
 
print("Train set Accuracy:",accuracy_score(y_train,kclassifier.predict(x_train)))
print("Train set Accuracy:",accuracy_score(y_test,y_pred))


# In[33]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# # Classification Report

# In[34]:


from sklearn.metrics import classification_report
report=classification_report(y_test,y_pred)
print(report)


# # Using SVM Approch
#    Building SVM Classifier

# In[35]:


from sklearn.svm import SVC
svc_classifier=SVC()
svc_classifier.fit(x_train,y_train)
y_pred2=svc_classifier.predict(x_test)
compare_df2 = pd.DataFrame({'Actual': y_test,'Predicted':y_pred})
compare_df2.sample(10)


# In[36]:


cnt=0
index=[]
for i,j in zip(compare_df2['Actual'],compare_df2['Predicted']):
    if(i!=j):
        cnt+=1
        print(cnt)


# In[37]:


print("Train set Accuracy:",accuracy_score(y_train,svc_classifier.predict(x_train)))
print("Train set Accuracy:",accuracy_score(y_test,y_pred2))


# In[38]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred2))


# In[39]:


from sklearn.metrics import classification_report
report=classification_report(y_test,y_pred2)
print(report)


# # Conclusion:
# SVM performed brtter than KNN Classifier.

# In[ ]:




