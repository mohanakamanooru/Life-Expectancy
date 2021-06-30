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

# In[19]:


# plotting the correleation between all the features in the dataset 
# to identify the strongly related related variables with LE

#Using Pearson Correlation
plt.figure(figsize=(20,15))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.show()


# In[20]:


# dataset["Life expectancy"] = pd.to_numeric(dataset["Life expectancy"])


# In[21]:


#Correlation with output variable Life expectancy
cor_target = correlation_matrix["Life expectancy"]

#Viewing highly correlated features
relevant_features = cor_target[cor_target>=-1]
relevant_features.sort_values(ascending=False)


# In[22]:


dataset.info()


# In[23]:


#DATA ENCODING - converting categorical and datetime variables to numeric 

#using label encoding for the categorical columns with text values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


dataset['Country'] = le.fit_transform(dataset['Country'])
dataset['Year'] = le.fit_transform(dataset['Year'])
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


#Correlation with output variable Life expectancy
cor_target = correlation_matrix["Life expectancy"]

#Selecting highly correlated features
relevant_features = cor_target[cor_target>=-1]
relevant_features.sort_values(ascending=False)


# In[28]:


#splitting the data into my train and test set 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


# In[29]:


#from sklearn import utils
#print(utils.multiclass.type_of_target(y_train))
#print(utils.multiclass.type_of_target(y_train.astype('int')))

y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[30]:


np.unique(y_train, return_counts=True)


# In[32]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_train,y_train = ros.fit_resample(X, y)

np.unique(y_train, return_counts=True)


# # 6. Applying Machine Learning Algorithms

# ### 6.1 KMEANS CLASSIFIER

# In[33]:


from sklearn.cluster import KMeans
import numpy as np

plt.rcParams["figure.figsize"] = (15,5)
kmeans = KMeans(n_clusters=4).fit(dataset)
#centroids = kmeans.cluster_centers_

plt.scatter(dataset['Life expectancy'],dataset['Eggs Consumption'],  c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)

plt.xlabel('Life expectancy')
plt.ylabel('Eggs Consumption')
plt.show()


# In[34]:


# visualising the countries consuming Egg comparitively much higher than other countries

raw_dataset[raw_dataset['Eggs Consumption']>18]['Country'].unique()


# In[35]:


from sklearn.cluster import KMeans
import numpy as np

plt.rcParams["figure.figsize"] = (15,5)
kmeans = KMeans(n_clusters=1).fit(dataset)
#centroids = kmeans.cluster_centers_

plt.scatter(dataset['Life expectancy'],dataset['Milk Consumption'],  c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)

plt.xlabel('Life expectancy')
plt.ylabel('Milk Consumption')
plt.show()


# In[36]:


# visualising the countries consuming Egg comparitively much higher than other countries

raw_dataset[raw_dataset['Milk Consumption']>350]['Country'].unique()


# In[37]:


from sklearn.cluster import KMeans
import numpy as np

plt.rcParams["figure.figsize"] = (15,5)
kmeans = KMeans(n_clusters=1).fit(dataset)

plt.scatter( dataset['Life expectancy'], dataset['Bovine Meat'],c= kmeans.labels_.astype(float), s=50, alpha=0.5)

plt.xlabel('Life expectancy')
plt.ylabel('Bovine Meat')

plt.show()


# In[38]:


# visualising the countries consuming Beef comparitively much higher than other countries

raw_dataset[raw_dataset['Bovine Meat']>40]['Country'].unique()


# In[39]:


from sklearn.cluster import KMeans
import numpy as np

plt.rcParams["figure.figsize"] = (15,5)
kmeans = KMeans(n_clusters=10).fit(dataset)

plt.scatter( dataset['Life expectancy'], dataset['Fish and Seafood'],c= kmeans.labels_.astype(float), s=50, alpha=0.5)

plt.xlabel('Life expectancy')
plt.ylabel('Fish and Seafood')

plt.show()


# In[40]:


# visualising the countries consuming fish and sea food comparitively much higher than other countries

raw_dataset[raw_dataset['Fish and Seafood']>75]['Country'].unique()


# ### 6.2 LOGISTIC REGRESSION

# In[41]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train,y_train)

#using the trained model on the test set
pred_y = model.predict(X_test)


print("Score on training set: {}".format(model.score(X_train,y_train)))
print("Score on test set: {}".format(model.score(X_test,y_test)))


# ### 6.3 RANDOMFOREST CLASSIFIER
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=50)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)


# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))


# ### 6.4 KNN CLASSIFIER 

# In[46]:


from sklearn.neighbors import KNeighborsClassifier

# checking the accuracy while looping throught the neighbors count from 5 to 20
for n in range(3,10):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print('Accuracy : {}'.format(knn.score(X_test, y_test)))


# In[47]:


from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

print('KNeighborsClassifier')
print('------------------------------')

# initialising the regressor for n=5
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


# In[48]:


#OBSERVATIONS:
    
#1. lONGER LOGEVITY --> Higher Egg and Meat Consumption ( fish and sea food - iceland , maldives
                        # Beef/ Bovine meat - Argentina )
#2. No effect of alcohol on logevity 
#3. Poultry and Goat meat has no effect too. 
#4. Higher Milk consumption shows higher longevity 


# In[ ]:




