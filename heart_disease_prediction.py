# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:33:35 2022
This script is to Predict if someone have Heart Disease using Machine Learning

1)Age : Age of the patient -->cont

2)Sex : Sex of the patient -->cat

3)exng: exercise induced angina (1 = yes; 0 = no) -->cat

4)oldpeak: ST depression induced by exercise relative to rest -->cont
    (‘ST’ relates to positions on the ECG plot)

5)slp: the slope of the peak exercise ST segment — 
    0: downsloping; 1: flat; 2: upsloping -->cat
   

6)caa: number of major vessels (0-3) -->cat

7)thall: A blood disorder called thalassemia -->cat
    Value 0: NULL (dropped from the dataset previously)
    Value 1: fixed defect (no blood flow in some part of the heart)
    Value 2: normal blood flow
    Value 3: reversible defect (a blood flow is observed but it is not normal)

8)cp : Chest Pain type chest pain type -->cat
    Value 1: typical angina
    Value 2: atypical angina
    Value 3: non-anginal pain
    Value 4: asymptomatic
    
9)trtbps : resting blood pressure (in mm Hg) -->cont

10)chol : cholestoral in mg/dl fetched via BMI sensor -->cont

11)fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)-->cat

12)restecg : resting electrocardiographic results -->cat
    Value 0: normal
    Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
    
13)thalachh : maximum heart rate achieved -->cont

14)output : 0= less chance of heart attack 1= more chance of heart attack -->cat

@author: User
"""

#%%
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

#%% Statics
CSV_PATH = os.path.join(os.getcwd(),'dataset','heart_train.csv')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'model','best_model_svc.pkl')

#%% Functions

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% EDA
# Step 1: Data Loading

df = pd.read_csv(CSV_PATH)

# Step 2: Data Inspection

df.info()
df_describe = df.describe().T

df.boxplot(figsize=(12,10))

df.isna().sum() # no NaN in dataset

df.duplicated().sum() # 1 duplicated data

# df[thall] contains nan(0=nan)
df['thall'] = df['thall'].replace(0,np.nan)

# To impute the value using mode
df['thall'].fillna(df['thall'].mode()[0], inplace=True)

# from the boxplot, there are outliers from:trtbps,chol,thalachh

## continuos data
cont_columns = ['age','trtbps','chol','thalachh','oldpeak']

for cont in cont_columns:
    plt.figure()
    sns.distplot(df[cont])
    plt.show()
    
## categorical data
cat_columns = ['sex','exng','slp','caa','cp','fbs','restecg','thall','output']

for cat in cat_columns:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()


# Step 3: Data Cleaning
## Remove duplicates 

df = df.drop_duplicates()
df.duplicated().sum()

X = df.drop(labels=['output'],axis=1)
y = df['output']

# Step 4: Features Selection

###categorical vs categorical: Cramer's V

print('Categorical Data: ')
for cat in cat_columns:
    print(cat)
    confussion_mat = pd.crosstab(df[cat],y).to_numpy()
    print(cramers_corrected_stat(confussion_mat))

# cp and thall been choosed for subsequent steps because 
# the correlation is hihgher than 0.5,
# 0.508955204032273 and 0.5206731262866439 respectively
    
###continuous vs categorical: Logistic Regression

lr = LogisticRegression()

print('Continuos Data: ')
for cont in cont_columns:
    lr.fit(np.expand_dims(df[cont],axis=-1),y)
    print(cont)
    print(lr.score(np.expand_dims(df[cont],axis=-1),y))
    
# from LR the age,thalachh and oldpeak been choosed for subsequent steps, 
# because the accuracy is above 60%

#Step 5 Preprocessing

# Train-test split

X = df.loc[:,['cp','thall','age','thalachh','oldpeak']]
y = df['output']

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=3)

#%% Pipeline
# 1) To determine whether MMS or SS is better in this case
# 2) To determine which classifier works the best in this case

#Logistic Regression
step_mms_lr = Pipeline([('MMS Scaler',MinMaxScaler()),
            ('LogisticClassifier',LogisticRegression())])

step_ss_lr = Pipeline([('SS Scaler',StandardScaler()),
            ('LogisticClassifier',LogisticRegression())])

#Random Forest
step_mms_rf = Pipeline([('MMS Scaler',MinMaxScaler()),
            ('RFClassifier',RandomForestClassifier())])

step_ss_rf = Pipeline([('SS Scaler',StandardScaler()),
            ('RFClassifier',RandomForestClassifier())])

#Decision TREE
step_mms_tree = Pipeline([('MMS Scaler',MinMaxScaler()),
            ('DTClassifier',DecisionTreeClassifier())])

step_ss_tree = Pipeline([('SS Scaler',StandardScaler()),
            ('DTClassifier',DecisionTreeClassifier())])

#KNN
step_mms_knn = Pipeline([('MMS Scaler',MinMaxScaler()),
            ('KNNClassifier',KNeighborsClassifier())])

step_ss_knn = Pipeline([('SS Scaler',StandardScaler()),
            ('KNNClassifier',KNeighborsClassifier())])

#SVC
step_mms_svc = Pipeline([('MMS Scaler',MinMaxScaler()),
            ('SVClassifier',SVC())])

step_ss_svc = Pipeline([('SS Scaler',StandardScaler()),
            ('SVClassifier',SVC())])

# Create Pipeline
pipelines = [step_mms_lr,step_ss_lr,
             step_mms_rf,step_ss_rf,
             step_mms_tree,step_ss_tree,
             step_mms_knn,step_ss_knn,
             step_mms_svc,step_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

#%% Model Evaluation/Pipeline Analysis

best_accuracy = 0

model_scored = []

for i, model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    model_scored.append(model.score(X_test,y_test))
    
best_pipeline = pipelines[np.argmax(model_scored)]
best_accuracy = model_scored[np.argmax(model_scored)]
        
print('The best combination of the pipeline is {} with accuracy {}'
      .format(best_pipeline.steps, best_accuracy))

#%% To fine tune the model

# GridSearchCV
# from the pipeline above, it is deduced that the pipeline with SS + SVC
# achieved the highest accuracy when tested against test dataset
step_svc = Pipeline([('SS Scaler',StandardScaler()),
            ('SVClassifier',SVC())])

grid_param = [{'SVClassifier': [SVC()],
              'SVClassifier__C': [1.0,10.0,100.0],
              'SVClassifier__kernel': ['linear','poly','sigmoid','rbf'],
              'SVClassifier__degree': [3,5,7],
              'SVClassifier__gamma':['scale','auto'],
              }]

grid_search = GridSearchCV(step_svc,grid_param,cv=5,verbose=1,
                           n_jobs=-1,error_score='raise')
best_model = grid_search.fit(X_train,y_train)

print('Best Model Score:')
print(best_model.score(X_test,y_test))
print('Best Model Index:')
print(best_model.best_index_)
print('Best Model Grid_Param:')
print(best_model.best_params_)

# Eventhough the various parameter used to fine tune the model but still the
# accuracy is not above 86% before tune the model, 
# the highest it can reach is between 83% to 85% using above parameters.
# Thus, the best model will be step_svc

#%% To save the best model
step_svc = Pipeline([('SS Scaler',StandardScaler()),
            ('SVClassifier',SVC())])

best_model_svc = step_svc.fit(X_train,y_train)

with open(BEST_MODEL_PATH, 'wb') as file:
    pickle.dump(best_model_svc,file)
        
#%% Model Analysis
# Classfication report, confusion matrix

y_true = y_test
y_pred = best_model_svc.predict(X_test)

print('Classification Report:')
print(classification_report(y_true,y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_true,y_pred))
print('Accuracy Score:')
print(accuracy_score(y_true,y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix(y_true,y_pred),
                              display_labels=step_svc.classes_)
disp.plot()
plt.show()

#%% Discussion
# The model is able give good prediction as the accuracy approximately 87%
# Eventhough we fine tune the model, but the accuracy is still not above 87%.
# During tuning the model with various combination of parameters, 
# the highest accuracy it can achieved is approximately 86% or
# no changes in the model eventhough we fine tune the model.
# When tested the model to test dataset, it can predict the output well.









