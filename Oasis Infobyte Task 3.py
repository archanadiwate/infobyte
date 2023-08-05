#!/usr/bin/env python
# coding: utf-8

# # Oasis Infobyte Data Science internship 2023
# Name - Archana Diwate

# # Task 3-EMAIL SPAM DETECTION WITH MACHINE LEARNING
#     We’ve all been the recipient of spam emails before. Spam mail, or junk mail, is a type of email that is sent to a massive number of users at one time, frequently containing cryptic messages, scams, or most dangerously, phishing content.
# In this Project, use Python to build an email spam detector. Then, use machine learning to train the spam detector to recognize and classify emails into spam and non-spam. Let’s get started!

# In[8]:


# Importing Required Libraries

import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import os

from warnings import filterwarnings
filterwarnings(action='ignore')


# In[9]:


df = pd.read_csv("spam.csv", encoding='latin-1')
df.head()


# In[10]:


df.tail()


# # Data Cleaning

# In[11]:


df.rename(columns = {'v1':'label', 'v2':'message'}, inplace = True)
df.head()


# In[12]:


df = df[['label','message']].copy()
df


# In[13]:


df.isna().sum()


# In[14]:


df.shape


# In[15]:


df.size


# In[16]:


df.info()


# In[17]:


df.describe()


# In[18]:


type(df)


# In[20]:


# Rename the Columns

df.rename({'v1':'Type','v2':'SMS'},axis=1,inplace=True)
df.head()


# In[21]:


## Checking Missing Values

df.isnull().sum()


# In[22]:


##Check for Duplicated Values

df.duplicated().sum()


# In[23]:


# Removing Duplicates

df = df.drop_duplicates(keep='first')
df.head()


# In[24]:


df.duplicated().sum()


# In[25]:


# To check spam and ham % of data plotting pie chart 

plt.pie(df['label'].value_counts(),labels=['ham','spam'],autopct='%0.2f%%',explode=[0.1,0])
plt.show()


# In[26]:


# Seems our data is imbalaced

# So doing Analysis on No. of Characters, Words, and Sentences Used in every Message

import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[28]:


## Number of Characters
df['num_characters'] = df['message'].apply(len) 
df.head()


# In[29]:


# Number of Words


df['num_words'] = df['message'].apply(lambda x:len(nltk.word_tokenize(x)))

df.head()


# In[30]:


df['num_sentences'] = df['message'].apply(lambda x:len(nltk.sent_tokenize(x)))

df.head()


# # Data Visualization

# In[31]:


#Plotting Spam(1) vs Not Spam(0) value counts using bar chart


df['label'].value_counts().plot(kind='bar')
plt.xlabel('label')
plt.ylabel('Count')
plt.show()   


# In[32]:


# Using Pairplot to identify any patterns, trends, or relationships between different features in a dataset

import seaborn as sns
sns.pairplot(df,hue='label')
plt.show()


# In[33]:


# Checking corr by using heat map 

sns.heatmap(df.corr(),annot=True)

plt.show()


# # Checking for Outliers

# In[34]:


# Using boxplot checking for outliers

plt.figure(figsize=(10,8))
sns.boxplot(x='label',y='num_characters',data=df)

plt.show()


# # Data Pre-Processing

# In[35]:


# Removing punctucation, stopwords, stemming.

from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def text_processing(text):
    text = nltk.word_tokenize(text.lower())
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            stemming = ps.stem(i)
            y.append(stemming)
            
    return " ".join(y)


# In[36]:


text_processing('I Loved the YT Lectures on machine Learning What About You! dacing dance danced')


# In[37]:


# Converting SMS text to Vectors by  Using Bag of Words Technique

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

tf = TfidfVectorizer(max_features=3000)


# In[38]:


# first convert label column to numeric (str to int )

df['label'] = df.label.map({'ham':0 , 'spam':1})

df.head()


# # Splitting data into Label and Features (X & Y)

# In[39]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.20, random_state=0)


# In[40]:


# Convert Sms To BOW Count verctor

count_vector = CountVectorizer()
train_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)


# In[41]:


# BOW(Bag Of Word) Look Like This 

# convert text to numbers for ML 

count_vector = CountVectorizer()
col_name = count_vector.fit(df['message']).get_feature_names()
data = count_vector.transform(list(df['message'])).toarray()
BOW = pd.DataFrame(data, columns= col_name)
BOW.head()


# # Using Navie Bayes

# In[42]:


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(train_data , y_train)


# In[43]:


predection = naive_bayes.predict(testing_data)


# In[44]:


from sklearn.metrics import accuracy_score ,f1_score , precision_score , recall_score

print('Accuracy score: {}'.format(accuracy_score(y_test, predection)))
print('precision_score: {}'.format(precision_score(y_test, predection)))
print('recall_score: {}'.format(recall_score(y_test, predection)))
print('f1_score: {}'.format(f1_score(y_test, predection)))


# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer
#removing stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string if there are nans
#dataset['description'] = dataset['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix_train = tfidf.fit_transform(X_train)
tfidf_matrix_valid= tfidf.transform(X_train)

tfidf_matrix_train.shape


# In[47]:


#Visualization to find the best K value
from sklearn.neighbors import KNeighborsClassifier
#To find the optimal k value: K=((Sqrt(N)/2)
#Visualisation for the Error Rate/K-value 
error_rate = []
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i, metric = 'minkowski', p=1)
    knn.fit(tfidf_matrix_train, y_train)
    pred_i_knn = knn.predict(tfidf_matrix_train)
    error_rate.append(np.mean(pred_i_knn != y_train))
plt.figure(figsize=(10,6))
plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# In[48]:


#Funtion to build and visualise a confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
def my_confusion_matrix(y_test, y_pred, plt_title, accuracy_title):
    cm=confusion_matrix(y_test, y_pred)
    print(f'{accuracy_title} Accuracy Score:', '{:.2%}'.format(accuracy_score(y_train, y_pred)))
    print(classification_report(y_test, y_pred))
    sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='magma')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(plt_title)
    plt.show()
    return cm


# In[49]:


X_train.isnull().any()


# In[50]:


#Fitting the KMM model
knn_classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p=1)
knn_classifier.fit(tfidf_matrix_train, y_train)
y_pred_knn=knn_classifier.predict(tfidf_matrix_valid)
cm_knn=my_confusion_matrix(y_train, y_pred_knn, 'KNN Confusion Matrix', 'KNN')


# In[51]:


#Training the model
from sklearn.linear_model import LogisticRegression
log_reg_classifier=LogisticRegression(solver='liblinear')
log_reg_classifier.fit(tfidf_matrix_train, y_train)
y_pred_log=log_reg_classifier.predict(tfidf_matrix_valid)
my_confusion_matrix(y_train, y_pred_log, 'Logistic Regression CM', 'Logistic Regression:')


# In[52]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(tfidf_matrix_train, y_train)
y_pred_rfc=rfc.predict(tfidf_matrix_valid)
print(my_confusion_matrix(y_train, y_pred_rfc, 'Random Forest', 'Random Forest'))


# In[53]:


from sklearn.svm import SVC
svc = SVC(kernel='rbf', C=10)
svc.fit(tfidf_matrix_train, y_train)
y_pred_svc= svc.predict(tfidf_matrix_valid)
cm_svc=my_confusion_matrix(y_train, y_pred_svc, 'Support Vector Classifier Confusion Matrix', 'SVC')


# In[ ]:


Thank you..

