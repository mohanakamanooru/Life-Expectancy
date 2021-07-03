#!/usr/bin/env python
# coding: utf-8

# # LIFE EXPECTANCY PROJECT
# 
# ####  Author : Mohana Kamanooru
# ####  Date : June 2021
# 

# # 1. Data Capture / Loading

# In[1]:


#import required libraries and pandas to read the raw data csv file to a dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the raw dataset 
raw_dataset = pd.read_csv('data/ConsolidatedDataV2.csv')


# # 2. Exploratory Data Analysis

# In[3]:


# The number of rows and columns in the dataset
raw_dataset.shape


# In[4]:


# Printing the first 6 rows of the dataset
raw_dataset.head()


# In[5]:


# Printing the last 6 rows of the dataset
raw_dataset.tail()


# In[6]:


# Count, Mean , Min and MAx values of the columns.
raw_dataset.describe()


# In[7]:


# To understand the data types of the columns
raw_dataset.info()


# # 3. Data Pre-processing

# In[8]:


dataset = raw_dataset


# In[9]:


## The dataset consists of “year” column which is time series data. So the datatype has been changed to datetime

dataset['Year'] = pd.to_datetime(dataset['Year'] , format='%Y', errors='ignore')
dataset.info()


# In[10]:


# Printing the first 6 rows of the dataset
dataset.head()


# In[11]:


#Total columns = 30
# Total rows = 2856

# to check the missing values in the columns and view in descending order
dataset.isnull().sum().sort_values(ascending=False)


# In[12]:


# Drop the columns with more than 50% of missing data 
# Cholera                2075
# Retirement Age         2074
# Measles                1227

#drop the retitrement column 
dataset = dataset.drop(['Retirement Age'],axis=1)

# removing the cholera column more than 50% missing data 
dataset = dataset.drop(['Cholera'],axis=1)

#drop the Measles column
dataset = dataset.drop(['Measles'],axis=1)


# In[13]:


# Life Expectancy is the main focus of our research 
# Life expectancy         289


# Imputing introduces bias which is not desirable . Hence, we delete this null rows 
dataset=dataset[dataset['Life expectancy'].notna()]


# In[14]:


# convert datatype of life expectancy to int after removing null values
dataset['Life expectancy'] = dataset['Life expectancy'].astype(np.int64)


# In[15]:


# to check the missing values in the columns and view in descending order again after removing some columns
dataset.isnull().sum().sort_values(ascending=False)


# In[16]:


# Drop only the rows with missing values which are less than 10% 

# Pig Meat               294
# Mutton & Goat meat     228
# Other Meat             228
# Fish and Seafood       228
# Milk Consumption       228
# Poultry Meat           228
# Eggs Consumption       228
# Bovine Meat            228
# Medical Expenditure    187
# Population             124
# Alcohol                 56
# Tuberculosis            38
# ChildMalnutrition       34
# Polio                   34
# Diphtheria              17

dataset=dataset[dataset['Pig Meat'].notna()]
dataset=dataset[dataset['Mutton & Goat meat'].notna()]
dataset=dataset[dataset['Other Meat'].notna()]

dataset=dataset[dataset['Fish and Seafood'].notna()]
dataset=dataset[dataset['Milk Consumption'].notna()]
dataset=dataset[dataset['Poultry Meat'].notna()]

dataset=dataset[dataset['Eggs Consumption'].notna()]
dataset=dataset[dataset['Bovine Meat'].notna()]
dataset=dataset[dataset['Medical Expenditure'].notna()]

dataset=dataset[dataset['Population'].notna()]
dataset=dataset[dataset['Alcohol'].notna()]
dataset=dataset[dataset['Tuberculosis'].notna()]

dataset=dataset[dataset['ChildMalnutrition'].notna()]
dataset=dataset[dataset['Polio'].notna()]
dataset=dataset[dataset['Diphtheria'].notna()]


# In[17]:


# replacing the missing values with mean value
# missing data is about 25% and deletion causes too much data loss from the dataset ( rows decrease drastically)

# HIV                     731
# HepatitisB              522
# BCG                     476

dataset['HIV']=dataset['HIV'].fillna(value=dataset['HIV'].mean())
dataset['BCG']=dataset['BCG'].fillna(value=dataset['BCG'].mean())
dataset['HepatitisB']=dataset['HepatitisB'].fillna(value=dataset['HepatitisB'].mean())


# # 4. Data Visualisation
# 
# ### visualising the data using dataprep library

# In[18]:


# To ignore warnings thrown by dataprep library 
import warnings
warnings.filterwarnings('ignore')

# importing the dataprep library
import dataprep
from dataprep.eda import create_report

# generating the visual using create_report method
pre_processed_report = create_report(dataset, title='Pre-Processed Dataset')
pre_processed_report


# # 5. Data Analysis

# ## 5.1 Label Encoding

# In[19]:


#using label encoding for the categorical columns with text values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


dataset['Country'] = le.fit_transform(dataset['Country'])
dataset['Year'] = le.fit_transform(dataset['Year'])
dataset.head()


# ## 5.2 Feature Selection / Analysing Correlation

# In[20]:


# plotting the correleation between all the features in the dataset 
# to identify the strongly related related variables with LE

#Using Pearson Correlation
plt.figure(figsize=(20,15))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.show()


# In[21]:


#Correlation with output variable Life expectancy
cor_target = correlation_matrix["Life expectancy"]

#Viewing highly correlated features
relevant_features = cor_target[cor_target>=-1]
relevant_features.sort_values(ascending=False)


# In[22]:


# dropping irrelevent features from the dataset 

# HepatitisB             0.244216
# BCG                    0.235935
# Year                   0.183378
# Mutton & Goat meat     0.056031
# Population             0.010906
# Country               -0.067061
# Other Meat            -0.090164
# Tuberculosis          -0.106261

dataset=dataset.drop('HepatitisB',axis=1)
dataset=dataset.drop('BCG',axis=1)
dataset=dataset.drop('Year',axis=1)
dataset=dataset.drop('Mutton & Goat meat',axis=1)
dataset=dataset.drop('Population',axis=1)
dataset=dataset.drop('Country',axis=1)
dataset=dataset.drop('Other Meat',axis=1)
dataset=dataset.drop('Tuberculosis',axis=1)


# In[23]:


dataset.head()


# In[24]:


# Feature Scaling

#transform the data to be on same scale using sklearn's StandardScaler()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()


X = dataset.drop('Life expectancy',axis=1)
y = dataset['Life expectancy'].astype('int')
X


# In[25]:


y


# In[26]:


X = scale.fit_transform(X)
X


# In[27]:


#splitting the data into my train and test set 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


# In[28]:


#from sklearn import utils
#print(utils.multiclass.type_of_target(y_train))
#print(utils.multiclass.type_of_target(y_train.astype('int')))

y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[29]:


np.unique(y_train, return_counts=True)


# In[30]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_train,y_train = ros.fit_resample(X, y)

np.unique(y_train, return_counts=True)


# # 6. Applying Machine Learning Algorithms

# In[68]:


from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from math import sqrt


# ### 6.0 Naive Bayes

# In[69]:


# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

print('Gaussian Naive Bayes')
print('------------------------------')

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print('Accuracy : {}'.format(gnb.score(X_test, y_test)))

MAE = mean_absolute_error(y_test, y_pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(y_test, y_pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(y_test, y_pred)
print('R2_SCORE  : %f' % R2_SCORE)

F1_SCORE = f1_score(y_test, y_pred, average='macro')
print('F1_SCORE  : %f' % F1_SCORE)


# In[56]:


# BernoulliNB Naive Bayes

from sklearn.naive_bayes import BernoulliNB

print('Bernoulli Naive Bayes')
print('------------------------------')

model = BernoulliNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy : {}'.format(model.score(X_test, y_test)))

MAE = mean_absolute_error(y_test, y_pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(y_test, y_pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(y_test, y_pred)
print('R2_SCORE  : %f' % R2_SCORE)

F1_SCORE = f1_score(y_test, y_pred, average='macro')
print('F1_SCORE  : %f' % F1_SCORE)


# ### 6.1 LOGISTIC REGRESSION

# In[34]:


from sklearn.linear_model import LogisticRegression

highAcc =0
maxc = 1
for c in range(1,20):
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c, 
                           fit_intercept=True, intercept_scaling=1, class_weight=None, 
                           random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', 
                           verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    model = model.fit(X_train,y_train)

    #using the trained model on the test set
    pred_y = model.predict(X_test)
    
    if(highAcc < model.score(X_test, y_test)):
        highAcc = model.score(X_test, y_test)
        maxc = c
        
print("C = {} ,  Accuracy = {}".format(maxc, highAcc))
   


# In[72]:



print('Logistic Regression , C=14')
print('------------------------------')

# initialising the classifier for c=14
model = LogisticRegression(C=14)

# applying the model for the test values
model = model.fit(X_train,y_train)

# predicting the out put values for test inputs 
y_pred = model.predict(X_test)

print('Accuracy : {}'.format(model.score(X_test, y_test)))

MAE = mean_absolute_error(y_test, y_pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(y_test, y_pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(y_test, y_pred)
print('R2_SCORE  : %f' % R2_SCORE)

F1_SCORE = f1_score(y_test, y_pred, average='macro')
print('F1_SCORE  : %f' % F1_SCORE)


# ### 6.2 KNN CLASSIFIER 

# In[36]:


from sklearn.neighbors import KNeighborsClassifier

print('KNeighborsClassifier')
print('------------------------------')
# checking the accuracy while looping throught the neighbors count from 5 to 10
highAcc =0 
maxn=0

for n in range(5,20):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    
    if(highAcc < knn.score(X_test, y_test)):
        highAcc = knn.score(X_test, y_test)
        maxn= n
        
    #print('n_neighbors {} -- Accuracy : {}'.format(n, knn.score(X_test, y_test) ))
print('n = {} , MaxAccuracy = {}'.format(maxn, highAcc ))


# In[58]:


print('KNeighborsClassifier')
print('------------------------------')

# initialising the classifier for n=5
knn = KNeighborsClassifier(n_neighbors = 5)

# applying the model for the test values
knn.fit(X_train,y_train)

# predicting the out put values for test inputs 
y_pred = knn.predict(X_test)

print('Accuracy : {}'.format(knn.score(X_test, y_test)))

MAE = mean_absolute_error(y_test, y_pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(y_test, y_pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(y_test, y_pred)
print('R2_SCORE  : %f' % R2_SCORE)

F1_SCORE = f1_score(y_test, y_pred, average='macro')
print('F1_SCORE  : %f' % F1_SCORE)


# ### 6.3 Support Vector Classification

# In[38]:


#SVC, NuSVC and LinearSVC are classes capable of performing binary and multi-class classification on a dataset. 

#Importing the necessary packages and libaries
from sklearn import svm


# In[39]:


print('SUPPORT VECTOR CLASSIFICATION ')
#print('------------------------------')

kernal = ["linear","rbf",'poly','sigmoid']

for k in kernal:
    #print('Kernal - {}'.format(k))
    print('------------------------------')
    for g in ["auto" ,"scale"]:
        highAcc = 0
        Maxc=1
        for c in range(1,15):
            # higher value of c gives l2 penality --> overfitting
            model = svm.SVC(C=c, kernel=k, gamma= g, decision_function_shape='ovo')
            model = model.fit(X_train,y_train)
            
            #using the trained model on the test set
            pred_y = model.predict(X_test)
            
            if highAcc < model.score(X_test,y_test):
                highAcc = model.score(X_test,y_test)
                Maxc = c

        print("Kernal = {} , C = {} , gamma = {} - MaxAccuracy = {}".format(k,Maxc,g,highAcc))
    #print('------------------------------')


# In[59]:


print('Linear SVC')
print('------------------------------')

# initialising the classifier
linear = svm.SVC(kernel='linear', C=10, decision_function_shape='ovo')

# applying the model for the test values
linear.fit(X_train,y_train)

# predicting the out put values for test inputs 
y_pred = linear.predict(X_test)

print('Accuracy : {}'.format(linear.score(X_test, y_test)))

MAE = mean_absolute_error(y_test, y_pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(y_test, y_pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(y_test, y_pred)
print('R2_SCORE  : %f' % R2_SCORE)

F1_SCORE = f1_score(y_test, y_pred, average='macro')
print('F1_SCORE  : %f' % F1_SCORE)


# In[60]:


print('Radial Basis Function - SVC')
print('------------------------------')

# initialising the classifier
rbf = svm.SVC(kernel='rbf', gamma="auto", C=14, decision_function_shape='ovo')

# applying the model for the test values
rbf.fit(X_train,y_train)

# predicting the out put values for test inputs 
y_pred = rbf.predict(X_test)

print('Accuracy : {}'.format(rbf.score(X_test, y_test)))

MAE = mean_absolute_error(y_test, y_pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(y_test, y_pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(y_test, y_pred)
print('R2_SCORE  : %f' % R2_SCORE)

F1_SCORE = f1_score(y_test, y_pred, average='macro')
print('F1_SCORE  : %f' % F1_SCORE)


# In[61]:


# polynomial kernel function 

poly = svm.SVC(kernel='poly', degree=3, C=14, gamma="auto", decision_function_shape='ovo').fit(X_train, y_train)

print('Polynomial Kernal Function - SVC')
print('------------------------------')

# predicting the out put values for test inputs 
y_pred = rbf.predict(X_test)

print('Accuracy : {}'.format(poly.score(X_test, y_test)))

MAE = mean_absolute_error(y_test, y_pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(y_test, y_pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(y_test, y_pred)
print('R2_SCORE  : %f' % R2_SCORE)

F1_SCORE = f1_score(y_test, y_pred, average='macro')
print('F1_SCORE  : %f' % F1_SCORE)


# In[62]:


#Sigmoid
sig = svm.SVC(kernel='sigmoid', C=5, gamma="scale", decision_function_shape='ovo').fit(X_train, y_train)


print('Sigmoid Function - SVC')
print('------------------------------')

# predicting the out put values for test inputs 
y_pred = rbf.predict(X_test)

print('Accuracy : {}'.format(sig.score(X_test, y_test)))

MAE = mean_absolute_error(y_test, y_pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(y_test, y_pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(y_test, y_pred)
print('R2_SCORE  : %f' % R2_SCORE)

F1_SCORE = f1_score(y_test, y_pred, average='macro')
print('F1_SCORE  : %f' % F1_SCORE)


# In[44]:


# To ignore warnings thrown by dataprep library 
import warnings
warnings.filterwarnings('ignore')

# importing the dataprep library
import dataprep
from dataprep.eda import create_report

# generating the visual using create_report method
final_report = create_report(dataset, title='Final Dataset')
final_report


# In[45]:


#OBSERVATIONS:
    
#1. lONGER LOGEVITY --> Higher Egg and Meat Consumption ( fish and sea food - iceland , maldives
                        # Beef/ Bovine meat - Argentina )
#2. No effect of alcohol on logevity 
#3. Poultry and Goat meat has no effect too. 
#4. Higher Milk consumption shows higher longevity 


# In[46]:


# MODEL                      Accuracy       MAE     MSE      RMSE      R2_Score    F1_SCORE
# Gaussian Naive Bayes         31%          1.63    5.96   2.440956    0.938516    0.249275
# Bernoulli Naive Bayes        22%          2.95    23.42  4.839478    0.758323    0.153571
# Logistic Regression          38%          1.16    3.83   1.958232    0.96043     0.340216
# KNN Classifier               85%          0.17    0.21   0.458555    0.99783     0.892837
# Linear SVC                   68%          0.4     0.58   0.763325    0.993987    0.703947
# Radial Basis Function SVC    82%          0.21    0.29   0.541978    0.996969    0.812503
# Polynomial SVC               79%          0.21    0.29   0.541978    0.996969    0.812503
# Sigmoid SVC                  29%          0.21    0.29   0.541978    0.996969    0.812503



# prefer Knn because of higher F1score , lower rmse , higher r2 score . 
# Though accuracy is not a measure to choose right model. KNN stands to be the best model for the current dataset.

