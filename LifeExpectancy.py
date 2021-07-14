#!/usr/bin/env python
# coding: utf-8

# # LIFE EXPECTANCY PROJECT
# 
# ####  Author : Mohana Kamanooru
# ####  Date : June 2021
# 

# # 1. Data Capture / Loading

# In[1]:


# Import required libraries and pandas to read the raw data csv file to a dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read the consolidated csv file into pandas dataframe 
RAW_DATASET = pd.read_csv('data/ConsolidatedDataV2.csv')


# # 2. Exploratory Data Analysis

# In[3]:


# Print the number of rows and columns in the raw dataset
RAW_DATASET.shape


# In[4]:


# Print the first 6 rows of the raw dataset
RAW_DATASET.head()


# In[5]:


# Print the last 6 rows of the raw dataset
RAW_DATASET.tail()


# In[6]:


# Generate descriptive statistics on raw dataset
RAW_DATASET.describe()


# In[7]:


# Print the data types of the columns in raw dataset
RAW_DATASET.info()


# # 3. Data Pre-processing

# In[8]:


# Creatine new dataframe called "dataset" to process and store the raw data 
dataset = RAW_DATASET


# In[9]:


# The dataset consists of “Year” column which is time series data. 

# TODO : Change data type of "Year" column to datetime
dataset['Year'] = pd.to_datetime(dataset['Year'] , format='%Y', errors='ignore')
dataset.info()


# In[10]:


# Print the first 6 rows of the dataset
dataset.head()


# In[11]:


# Total columns = 30
# Total rows = 2856

# TODO : check the count of missing values by columns in descending order
dataset.isnull().sum().sort_values(ascending=False)


# In[12]:


# TODO : Drop the columns with more than 50% of missing data 

# Cholera                2075
# Retirement Age         2074
# Measles                1227

# Drop the cholera column more than 50% missing data 
dataset = dataset.drop(['Cholera'],axis=1)

# Drop the retitrement column 
dataset = dataset.drop(['Retirement Age'],axis=1)

# Drop the Measles column
dataset = dataset.drop(['Measles'],axis=1)


# In[13]:


# Life Expectancy is the main focus of our research
# Imputing introduces bias which is not desirable . Hence, we delete this null rows 
# Life expectancy         289

# Filter the rows with null values "Life expectancy" column of dataset
dataset=dataset[dataset['Life expectancy'].notna()]


# In[14]:


# Convert datatype of life expectancy to int after removing null values
# To apply classification algorithms

dataset['Life expectancy'] = dataset['Life expectancy'].astype(np.int64)


# In[15]:


# TODO : check the count of missing values by columns in descending order after removing some columns and rows
dataset.isnull().sum().sort_values(ascending=False)


# In[16]:


# TODO : Drop only the rows with missing values which are less than 10% 

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


# TODO : Replace the missing values with mean value
# missing data is about 25% and deletion causes too much data loss from the dataset ( rows decrease drastically)

# HIV                     731
# HepatitisB              522
# BCG                     476

dataset['HIV']=dataset['HIV'].fillna(value=dataset['HIV'].mean())
dataset['BCG']=dataset['BCG'].fillna(value=dataset['BCG'].mean())
dataset['HepatitisB']=dataset['HepatitisB'].fillna(value=dataset['HepatitisB'].mean())


# # 4. Data Visualisation
# 
# ### Visualising the data using dataprep library

# In[18]:


# Ignore and supress the warnings thrown by dataprep library 
import warnings
warnings.filterwarnings('ignore')


# In[19]:


# Import the dataprep library
import dataprep

from dataprep.eda import create_report

# Generate the visual report called "pre_processed_report" using create_report method
pre_processed_report = create_report(dataset, title='Pre-Processed Dataset')
pre_processed_report


# # 5. Data Analysis

# ## 5.1 Label Encoding

# In[20]:


# Using label encoder for the categorical columns with text values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# TODO : Encode labels "Country" and "Year"
dataset['Country'] = le.fit_transform(dataset['Country'])
dataset['Year'] = le.fit_transform(dataset['Year'])
dataset.head()


# ## 5.2 Feature Selection / Analysing Correlation

# In[21]:


# Plot the correleation between all the features in the dataset 
# Identify the strongly related related variables with Life Expectancy

#Using Pearson Correlation
plt.figure(figsize=(20,15))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.show()


# In[22]:


# Correlation with output variable Life expectancy
cor_target = correlation_matrix["Life expectancy"]

# View highly correlated features
relevant_features = cor_target[cor_target>=-1]
relevant_features.sort_values(ascending=False)


# In[23]:


# Drop features with no strong correlation with the target variable from the dataset 

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


# In[24]:


# Print first 6 rows in dataset 
dataset.head()


# In[25]:


# TODO : Feature Scaling

#transform the data to be on same scale using sklearn's StandardScaler()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()


X = dataset.drop('Life expectancy',axis=1)
y = dataset['Life expectancy'].astype('int')

# Print the features "X" and target variable "y"
print( X , y)


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

# In[31]:


from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from math import sqrt


# ### 6.1 LOGISTIC REGRESSION

# In[32]:


from sklearn.linear_model import LogisticRegression

#defining default values to store high accuracy and values of C at higher ccuracy
HIGH_ACCURACY =0
MAX_C = 1

# Looping for values of c from 1 till 20 to check the behaviour and accuracy of the model.
for c in range(1,20):
    
    #defining linear model with default parameters and C=c
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c, 
                           fit_intercept=True, intercept_scaling=1, class_weight=None, 
                           random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', 
                           verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    
    #Fitting the model for the test and train sets
    model = model.fit(X_train,y_train)

    #Predicting the test values with the built model 
    pred_y = model.predict(X_test)
    
    #printing the accuracy on the console
    print("C = {} ,  Accuracy = {}".format(c, model.score(X_test, y_test)))
    
    #if accuracy is high then store it in the variables
    if(HIGH_ACCURACY < model.score(X_test, y_test)):
        HIGH_ACCURACY = model.score(X_test, y_test)
        MAX_C = c
        
    
#Printing the the highest accuracy achieved and the respective C value     
print("*************************************")
print("Maximum Accuracy is achieved at C = {} ,  Accuracy = {}".format(MAX_C, HIGH_ACCURACY))
   


# In[33]:


# DEFINING MODEL WITH PARAMETER C=14 (HIGH ACCURACY)
model = LogisticRegression(C=14)

# TRAINING THE DEFINED MODEL
model = model.fit(X_train,y_train)

# PREDICTING THE TARGET WITH THE TRAINED MODEL
y_pred = model.predict(X_test)

# CALCULATING MEAN ABSOLUTE ERROR
MAE = mean_absolute_error(y_test, y_pred)

# CALCULATING THE MEAN SQUARED ERROR
MSE = mean_squared_error(y_test, y_pred)

# CALCULATING ROOT MEAN SQUARED ERROR
RMSE = sqrt(MSE)

# CALCULATING R2_SCORE 
R2_SCORE=r2_score(y_test, y_pred)

# CALCULATING F1_SCORE 
F1_SCORE = f1_score(y_test, y_pred, average='macro')

#PRINTING THE CALCULATED METRICS
print('LOGISTIC REGRESSION , C=14')
print('------------------------------')
# PRININTING THE ACCURACY OF THE MODEL WITH DEFINED C VALUE
print('Accuracy : {}'.format(model.score(X_test, y_test)))
print('MAE : {}'.format(round(MAE, 2)))
print('MSE : {}'.format(round(MSE, 2)))
print('RMSE  : %f' % RMSE)
print('R2_SCORE  : %f' % R2_SCORE)
print('F1_SCORE  : %f' % F1_SCORE)


# ### 6.2 NAIVE BAYES

# In[34]:


# GAUSSIAN NAIVE BAYES

#IMPORTING GAUSSIAN NAIVE BAYES PACKAGE
from sklearn.naive_bayes import GaussianNB

# DEFINING MODEL
gnb = GaussianNB()

# TRAINING THE DEFINED MODEL AND PREDICTING THE TARGET WITH TRAINED MODEL
y_pred = gnb.fit(X_train, y_train).predict(X_test)

# CALCULATING MEAN ABSOLUTE ERROR
MAE = mean_absolute_error(y_test, y_pred)

# CALCULATING THE MEAN SQUARED ERROR
MSE = mean_squared_error(y_test, y_pred)

# CALCULATING ROOT MEAN SQUARED ERROR
RMSE = sqrt(MSE)

# CALCULATING R2_SCORE 
R2_SCORE=r2_score(y_test, y_pred)

# CALCULATING F1_SCORE 
F1_SCORE = f1_score(y_test, y_pred, average='macro')

print('GAUSSIAN NAIVE BAYES ')
print('------------------------------')
print('Accuracy : {}'.format(gnb.score(X_test, y_test)))
print('MAE : {}'.format(round(MAE, 2)))
print('MSE : {}'.format(round(MSE, 2)))
print('RMSE  : %f' % RMSE)
print('R2_SCORE  : %f' % R2_SCORE)
print('F1_SCORE  : %f' % F1_SCORE)


# In[35]:


# BernoulliNB Naive Bayes
from sklearn.naive_bayes import BernoulliNB

# DEFINING MODEL
bnb = BernoulliNB()

# TRAINING THE DEFINED MODEL AND PREDICTING THE TARGET WITH TRAINED MODEL
model = bnb.fit(X_train, y_train).predict(X_test)

# CALCULATING MEAN ABSOLUTE ERROR
MAE = mean_absolute_error(y_test, y_pred)

# CALCULATING THE MEAN SQUARED ERROR
MSE = mean_squared_error(y_test, y_pred)

# CALCULATING ROOT MEAN SQUARED ERROR
RMSE = sqrt(MSE)

# CALCULATING R2_SCORE 
R2_SCORE=r2_score(y_test, y_pred)

# CALCULATING F1_SCORE 
F1_SCORE = f1_score(y_test, y_pred, average='macro')

print('BERNOULLI NAIVE BAYES ')
print('------------------------------')
print('Accuracy : {}'.format(bnb.score(X_test, y_test)))
print('MAE : {}'.format(round(MAE, 2)))
print('MSE : {}'.format(round(MSE, 2)))
print('RMSE  : %f' % RMSE)
print('R2_SCORE  : %f' % R2_SCORE)
print('F1_SCORE  : %f' % F1_SCORE)


# ### 6.3 KNN CLASSIFIER 

# In[36]:


from sklearn.neighbors import KNeighborsClassifier

#defining default values to store high accuracy and values of n at higher ccuracy
HIGH_ACCURACY =0
MAX_N = 1

# Looping for values of n from 5 till 20 to check the behaviour and accuracy of the model.
for n in range(5,20):
    
    #defining model with default parameters and n_neighbors = n
    knn = KNeighborsClassifier(n_neighbors = n)
    
    #Fitting the model for the test and train sets
    knn.fit(X_train,y_train)
    
     #Predicting the test values with the built model
    y_pred = knn.predict(X_test)
    
    #printing the accuracy on the console
    print('n = {} , Accuracy = {}'.format(n, knn.score(X_test, y_test)))
    
     #if accuracy is high then store it in the variables
    if(HIGH_ACCURACY < knn.score(X_test, y_test)):
        HIGH_ACCURACY = knn.score(X_test, y_test)
        MAX_N= n
        
    
# print('KNeighborsClassifier')
#Printing the the highest accuracy achieved and the respective C value     
print("*************************************")
print("Maximum Accuracy is achieved at n = {} ,  Accuracy = {}".format(MAX_N, HIGH_ACCURACY))


# In[37]:


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


# ### 6.4 Support Vector Classification

# In[38]:


#SVC, NuSVC and LinearSVC perform both binary and multi-class classification on a dataset. 


# In[39]:


#Importing the necessary packages and libaries
from sklearn import svm

print('SUPPORT VECTOR CLASSIFICATION ')

# LIST OF KERNALS
kernal = ["linear","rbf",'poly','sigmoid']

#LOOPING THROUGH THE LIST OF KERNALS
for k in kernal:
   
    print('------------------------------')
    
    # LOOPING THROUGH THE PARAMETER g TO TUNE OUR MODEL ACCORDINGLY 
    for g in ["auto" ,"scale"]:
        HIGH_ACC = 0
        Max_C = 1
        
        # LOOPING THROUGH VALUE OF C FROM 1 TO 15
        # higher value of c gives l2 penality --> overfitting
        for c in range(1,15):
            
            # DEFINING THE MODEL
            model = svm.SVC(C=c, kernel=k, gamma= g, decision_function_shape='ovo')
            
            # TRAINING THE CREATED MODEL
            model = model.fit(X_train,y_train)
            
            # PREDICTING THE TATRGET WITH TEST VALUES
            pred_y = model.predict(X_test)
            
            # STORING THE HIGH ACCURACY AND CORRESPONDING C AND g VALUES
            if HIGH_ACC < model.score(X_test,y_test):
                HIGH_ACC = model.score(X_test,y_test)
                Max_C = c

        # PRINTING THE HIGH ACCURACY FOR ALL THE KERNAL WITH DIFF PARAMETER COMBINATIONS
        print("Kernal = {} , C = {} , gamma = {} - MaxAccuracy = {}".format(k,Max_C,g,HIGH_ACC))


# In[40]:


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


# In[41]:


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


# In[42]:


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


# In[43]:


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


# # 7 Conclusions

# ## 7.1 Protein Consumption Vs Life Expectancy

# In[45]:


# To ignore warnings thrown by dataprep library 
import warnings
warnings.filterwarnings('ignore')

# importing the dataprep library
import dataprep
from dataprep.eda import create_report

# generating the visual using create_report method
final_report = create_report(dataset, title='Final Dataset')
final_report


# In[47]:


#creating a new dataset with life expectancy greater than 73
countries_above73 = RAW_DATASET[RAW_DATASET['Life expectancy']>73]


# In[48]:


# Listing the countries with high consumption of egg
countries_above73[countries_above73['Eggs Consumption']>18]['Country'].unique()


# In[49]:


# Listing the countries with high consumption of beef
countries_above73[countries_above73['Bovine Meat']>35]['Country'].unique()


# In[50]:


# Listing the countries with high consumption of Pork
countries_above73[countries_above73['Pig Meat']>50]['Country'].unique()


# In[51]:


# Listing the countries with high consumption of Poultry Meat
countries_above73[countries_above73['Poultry Meat']>50]['Country'].unique()


# In[52]:


# Listing the countries with high consumption of Milk
countries_above73[countries_above73['Milk Consumption']>300]['Country'].unique()


# In[53]:


# Listing the countries with high consumption of seafood
countries_above73[countries_above73['Fish and Seafood']>50]['Country'].unique()


# In[54]:


# Grouping protein consumption columns 
# Protein_intake = Eggs Consumption + Bovine Meat + Pig Meat + Poultry Meat + Poultry Meat + Fish and Seafood

dataset["Protein_Intake"] = dataset["Eggs Consumption"] + dataset["Bovine Meat"] + dataset["Pig Meat"] + dataset["Poultry Meat"] + dataset["Poultry Meat"] + dataset["Fish and Seafood"]

dataset.head()


# In[55]:


plt.rcParams["figure.figsize"] = (15,5)
plt.scatter(dataset["Life expectancy"], dataset["Protein_Intake"], alpha=0.5)
plt.xlabel('Life expectancy')
plt.ylabel('Protein_Intake')
plt.show()


# In[ ]:




