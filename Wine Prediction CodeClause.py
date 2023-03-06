#!/usr/bin/env python
# coding: utf-8

#                                                    *CODECLAUSE* 
#                                               Wine Quality Prediction
# 
# 
# 
# ## Business Problem : 
# 
#    *- Wine Quaity Prediction using Machine Learning
#     
# ### Business objective :
# 
#    *- Minimise - Alcohol Content,
#                  Unpleasant Taste,
#                  Contamination
#                  
# ## Business Constraints
# 
#    *- Maximise - Wine Making Process,
#                  Grape variety & Wineyard Management,
#                  Aging Process,
#                  Fermentation Process,

# * Project Description - 
# 
#     - On the weekend, most of us prefer having a fancy dinner with our loved ones. While the kids define a fancy dinner as one that has pasta, adults like to add a cherry on top by having a classic glass of red wine along with the Italian dish. But when it comes to shopping for that wine bottle, a few of us get confused about which is the best one to buy. Few believe that the longer it has been fermented, the better it'll taste. Few suggest relatively sweeter wines are good quality wines. To know a

# In[4]:


import pandas as pd 
import numpy as np
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train = pd.read_csv("train.csv")
train


# In[6]:


test = pd.read_csv("test.csv")
test


# ## First Business Moment

# In[7]:


train.info()


# In[8]:


test.info()


# In[ ]:


## Performing Exploratory Data Analysis for Train & Test Datasets as added as DF & DT


# In[10]:


from pandas_profiling import ProfileReport
profile = ProfileReport(train, title = 'Pandas Profiling Report')
profile


# In[11]:


from pandas_profiling import ProfileReport
profile = ProfileReport(test, title = 'Pandas Profiling Report')
profile


# In[12]:


train.shape


# In[13]:


test.shape


# In[14]:


train.describe()


# In[15]:


test.describe()


# In[16]:


## Univariate Analysis
### To check the frequency of Each Column we Perform Histogram


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


## Histogram

features_category = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','Id']
for feature in features_category:
  plt.hist(data=train, x=feature)
  plt.xticks(rotation=90)
  plt.title(f'Histogram of {feature}')
  plt.show()


# In[21]:


## Histogram

features_category = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','Id']
for feature in features_category:
  plt.hist(data=test, x=feature)
  plt.xticks(rotation=90)
  plt.title(f'Histogram of {feature}')
  plt.show()


# ## Generated each features for Histogram. It helps to visualise the distribution of eeach categorical feature.The purpose of this code is to explore the data and gain insights into the distribution of each feature.

# ## The correlation matrix is a table that shows the correlation coefficients between variables at the intersection of the corresponding rows and columns.

# In[22]:


corrmat = train.corr()
print(corrmat)


# In[23]:


corrmat = test.corr()
print(corrmat)


# In[26]:


corrmat = train.corr(method='pearson')

plt.figure(figsize=(15, 10))
sns.heatmap(corrmat, annot=True)
plt.show()


# In[27]:


corrmat = test.corr(method='pearson')

plt.figure(figsize=(15, 10))
sns.heatmap(corrmat, annot=True)
plt.show()


# ## Attempting to predict the quality level of wine based on other aspects of the wines. This type of analysis shows that we could estimate any wine characteristic from the others, as all variables have a significant correlation rate with several others.
# 
# However, it should be noted that analyzing correlations between variables is interesting but not sufficient to make relevant predictions. Indeed, a linear regression of the form below would be more appropriate: quality = avariable1 + bvariable2 + ... + n*variablen.
# 
# A linear regression allows us to assess the degree of accuracy of the correlation using a p-value. A p-value below certain thresholds allows us to assert that the correlation between two variables is relevant.

# ## Correlations with the target variable.

# In[28]:


## We are looking at correlations with our target criterion: wine quality.

corrmat = train["quality"].sort_values(ascending = False)
print(corrmat)


# The alcohol, sulfate, and citric acid levels appear to be directly related to wine quality. Fixed acidity and residual sugar also have a positive linear relationship with quality, but their impact is weak. On the other hand, pH, sulfur dioxide, density, and volatile acidity are negatively correlated with wine quality. (It is not pertinent to interpret the correlation with the Id. It should be excluded from the training algorithm to avoid bias.)

# In[52]:


train


# ## Model Building 
# To develop the Alogrithms that can make predictions or classify new data based on patterns found in existing data.
# Its helps us for Prediction, Classifiction, Optimization, Insights.

# In[ ]:


#considering model building for train dataset 


# In[62]:


# separating the data and labels
X_train = train.drop(columns = ['Output','quality','Id'],axis=1)
Y_test = train['Output']


# In[63]:


X_train


# In[64]:


Y_test


# In[69]:


X_test = test.drop(columns = 'Id',axis=1)


# In[70]:


X_test


# In[68]:


print(X_train.shape)


# In[71]:


print(X_test.shape)


# In[72]:


# splitting the data into testing and training data.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify = Y)


# ### Support Vectoor Machine is a supervised machine learning algorithm that can be used for classification and regression tasks. It is based on the idea of finding the hyperplane that maximizes the margin between different classes of data points.

# In[75]:


# Support Vector Machine (SVM-Black Box Technique)   
from sklearn import svm
from sklearn.metrics import accuracy_score
classifier = svm.SVC(kernel='linear')


# In[76]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[77]:


# Accuracy score on the training data

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)


# In[78]:


# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


# Accuracy Rate For SVM :
#     Train - 0.861
#     Test - 0.860

# ### Logistic Regression - The ML Algorithm used for classifiction tasks. It is used to predict the probability of Binary Outcome

# In[79]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# In[80]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# In[81]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[82]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[83]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[84]:


print('Accuracy on Test data : ', test_data_accuracy)


# Accuracy Rate For Logistic Regression :
#     Train - 0.878
#     Test - 0.886

# ### Decision Tree -  The idea of creating a tree-like model of decisions and their possible consequences. Also heps us in predicting the model. It can handle both continuos & categorical outputs.

# In[85]:


#Desicion tree Model

#Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score,confusion_matrix

# Model building for Decision Tree

dtc = DecisionTreeClassifier(criterion="gini", max_depth=3)

dtc.fit(X_train,Y_train)
Y_train_pred = dtc.predict(X_train)
Y_test_pred = dtc.predict(X_test)

print(f'Train score {accuracy_score(Y_train_pred,Y_train)}') 
print(f'Test score {accuracy_score(Y_test_pred,Y_test)}') 

# confusion matrix for performance metrics
cm = confusion_matrix(Y_test, Y_test_pred)
cm
# or
pd.crosstab(Y_test, Y_test_pred, rownames=['Actual'], colnames=['Predictions'])

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_test_pred))


# Accuracy Rate for Decision Tree 
#     Train - 0.891
#     Test  - 0.877

# In[86]:


# Training the model Random forest model
from sklearn.ensemble import RandomForestClassifier

classifier_rfg=RandomForestClassifier(random_state=33,n_estimators=23)
parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}]

model_gridrf=GridSearchCV(estimator=classifier_rfg, param_grid=parameters, scoring='accuracy',cv=10)
model_gridrf.fit(X_train,Y_train)

model_gridrf.best_params_

# Predicting the model
Y_predict_rf = model_gridrf.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(Y_test,Y_predict_rf))
print(classification_report(Y_test,Y_predict_rf))


# In[ ]:


Accuracy Rate for Random Forest Classifier :
    Train - 0.90
    Test - 0.91


# In[88]:


## Building a Predictive System 
input_data = (7.5,0.500,0.36,6.1,0.071,17.0,102.0,0.99780,3.35,0.80,10.5)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model_gridrf.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('Wine Quality Is Bad')
else:
  print('Wine Quality Is Good')


# ## By this model we can understand that The Random Forest Classifier is the best model of built one. 
# 
#        - We can predict the model again by creating the Web App by using various Cloud Platforms & We can also Predict it by Using Predictive System with Encoded Variables which gives us the Output of Wine Quality, considering How the quality of Wine.

# In[92]:


import os 
os.getcwd()


# In[ ]:




