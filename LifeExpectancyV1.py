#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas to read the raw data csv file to a dataframe
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the raw dataset 
dataset = pd.read_csv('data/ConsolidatedDataV2.csv')


# To understand the number of rows and columns in the dataset
dataset.shape


# In[3]:


# Printing the first 6 rows of the dataset
dataset.head()


# In[4]:


# Printing the last 6 rows of the dataset
dataset.tail()


# In[5]:


# min, max count avg and percentile details of each column 
dataset.describe()


# In[6]:


# To understand the data types of the column data
dataset.info()


# In[7]:


import warnings
warnings.filterwarnings('ignore')

import dataprep

from dataprep.eda import create_report

report = create_report(dataset, title='My Report')
report


# In[8]:


# filling missing values with world avg retirement age 
dataset['Retirement Age']=pd.to_numeric(dataset['Retirement Age'].fillna(value="66"))

# removing the cholers column more than 50% missing data 
dataset=dataset[dataset['Cholera'].notna()]

# filling missing values with world avg life expectancy
dataset['Life expectancy']=dataset['Life expectancy'].fillna(value="72.6")

# filling missing values with mean values
dataset['BCG']=dataset['BCG'].fillna(value=dataset['BCG'].mean())
dataset['HIV']=dataset['HIV'].fillna(value="100") # less than 100 
dataset['Population']=dataset['Population'].fillna(value=dataset['Population'].mean())
dataset['Eggs Consumption']=dataset['Eggs Consumption'].fillna(value=dataset['Eggs Consumption'].mean())
dataset['Bovine Meat']=dataset['Bovine Meat'].fillna(value=dataset['Bovine Meat'].mean())
dataset['Mutton & Goat meat']=dataset['Mutton & Goat meat'].fillna(value=dataset['Mutton & Goat meat'].mean())
dataset['Other Meat']=dataset['Other Meat'].fillna(value=dataset['Other Meat'].mean())
dataset['Pig Meat']=dataset['Pig Meat'].fillna(value=dataset['Pig Meat'].mean())
dataset['Poultry Meat']=dataset['BCG'].fillna(value=dataset['Poultry Meat'].mean())
dataset['Milk Consumption']=dataset['Milk Consumption'].fillna(value=dataset['Milk Consumption'].mean())
dataset['Medical Expenditure']=dataset['Medical Expenditure'].fillna(value=dataset['Medical Expenditure'].mean())
dataset['HepatitisB']=dataset['HepatitisB'].fillna(value=dataset['HepatitisB'].mean())
dataset['Alcohol']=dataset['Alcohol'].fillna(value=dataset['Alcohol'].mean())
dataset['ChildMalnutrition']=dataset['ChildMalnutrition'].fillna(value=dataset['ChildMalnutrition'].mean())
dataset['Measles']=dataset['Measles'].fillna(value=dataset['Measles'].mean())
dataset['Polio']=dataset['Polio'].fillna(value=dataset['Polio'].mean())
dataset['Tuberculosis']=dataset['Tuberculosis'].fillna(value=dataset['Tuberculosis'].mean())


# In[9]:


# convert datatype of HIV, Life expectancy columns 
dataset["HIV"] = pd.to_numeric(dataset["HIV"])

dataset["Life expectancy"] = pd.to_numeric(dataset["Life expectancy"])


# In[10]:


report = create_report(dataset, title='Cleaned_report')
report


# In[11]:


dataset.info()


# In[12]:


#removing the Column Country  ( check duplicated after dropping country)
dataset = dataset.drop(['Country'],axis=1)


# In[13]:


# checking duplicates 
report = create_report(dataset, title='preprocessed_data')
report

#DATA ENCODING 

#using label encoding for the categorical columns with text values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


dataset['Country'] = le.fit_transform(dataset['Country'])
dataset.head()
# In[14]:


dataset.info()


# In[15]:


# Feature Scaling

#transform the data to be on same scale using sklearn's StandardScaler()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X = dataset.drop('Life expectancy',axis=1)
y = dataset['Life expectancy'].astype('int')
X


# In[16]:


y


# In[17]:


X = scale.fit_transform(X)
X


# In[18]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Using Pearson Correlation
plt.figure(figsize=(20,15))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.show()


# In[19]:


#Correlation with output variable module score
cor_target = correlation_matrix["Life expectancy"]

#Selecting highly correlated features
relevant_features = cor_target[cor_target>=-1]
relevant_features.sort_values(ascending=False)


# In[20]:


#splitting the data into my train and test set 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


# In[21]:


#from sklearn import utils
#print(utils.multiclass.type_of_target(y_train))
#print(utils.multiclass.type_of_target(y_train.astype('int')))

y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[22]:


np.unique(y_train, return_counts=True)


# In[23]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_train_sm,y_train_sm = ros.fit_resample(X, y)

#sampler = SMOTE(random_state=42)
#X_train_sm,y_train_sm = sampler.fit_resample(X,y)

np.unique(y_train_sm, return_counts=True)


# In[24]:


X_train = X_train_sm
y_train = y_train_sm


# In[39]:


from sklearn.cluster import KMeans
import numpy as np

plt.rcParams["figure.figsize"] = (15,5)
kmeans = KMeans(n_clusters=6).fit(dataset)

plt.scatter(dataset['Eggs Consumption'],dataset['Life expectancy'],  c= kmeans.labels_.astype(float), s=100, alpha=0.5)
plt.ylabel('Life expectancy')
plt.xlabel('Eggs Consumption')
plt.show()


# In[26]:


from sklearn.cluster import KMeans
import numpy as np

plt.rcParams["figure.figsize"] = (15,5)
kmeans = KMeans(n_clusters=1).fit(dataset)

plt.scatter( dataset['Life expectancy'], dataset['Bovine Meat'],c= kmeans.labels_.astype(float), s=100, alpha=0.5)

plt.xlabel('Life expectancy')
plt.ylabel('Bovine Meat')

plt.show()


# In[27]:


from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)


# In[28]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[29]:


from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))


# In[30]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train,y_train)

#using the trained model on the test set
pred_y = model.predict(X_test)


print("Score on training set: {}".format(model.score(X_train,y_train)))
print("Score on test set: {}".format(model.score(X_test,y_test)))


# In[31]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt


print('KNeighborsClassifier')
print('------------------------------')
# initialising the regressor for n=3
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


# In[32]:


from sklearn.neighbors import KNeighborsClassifier

# checking the accuracy while looping throught the neighbors count from 5 to 20
for n in range(3,20):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print('Accuracy : {}'.format(knn.score(X_test, y_test)))


# In[33]:


#OBSERVATIONS:
    
#1. lONGER LOGEVITY --> Higher Egg and Meat Consumption
#2. No effect of alcohol on logevity 
#3. Poultry and Goat meat has no effect too. 
#4. Higher Milk consumption shows higher longevity 

