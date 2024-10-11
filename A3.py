#==========================In the name of god =================================

# import lib section 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================



#------------------------------------------------------------------------------
#-------------------->>>> import Data (breast_cancer) <<<<<--------------------
#------------------------------------------------------------------------------


from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()


#------------------------------------------------------------------------------
#-------------------------->>>>  x,y  <<<<<------------------------------------
#------------------------------------------------------------------------------

x=data.data
y=data.target


print(data.feature_names)

'''
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
'''
print(data.target_names)

'''
['malignant' 'benign']
'''

x=pd.DataFrame(x, columns=data.feature_names)

y=pd.DataFrame(y,columns=['target'])


#------------------------------------------------------------------------------
#----------------------->>>> Step 0, Data Cleaning  <<<<<----------------------
#------------------------------------------------------------------------------

x.info()
x.describe()


'''
#Empty cell
x_cleaned= x.dropna()
cc=len(x)-len(x_cleaned)
print('There is ',cc, ' empty cell in the table') #There is  0  empty cell in the table



#duplicates 
new_x=x.drop_duplicates()
nn=len(x)-len(new_x)
print('There is ',nn, ' duplicated-data in the table') #There is  0  duplicated-data in the table
'''

#------------------------------------------------------------------------------
#----------------------->>>> Step 2, CV determination  <<<<<-------------------
#------------------------------------------------------------------------------

from sklearn.model_selection import KFold

#568 data --->  test dataset=71  --->  kf=568/71= 8
kf=KFold(n_splits=8, shuffle=True, random_state=42)


#------------------------------------------------------------------------------
#----------------------->>>> Step 3, Model selection  <<<<<--------------------
#------------------------------------------------------------------------------


#~~~~~~~~~~~~~~>>>>>>> Model: LogosticRegression <<<<<<<<~~~~~~~~~~~~~~~~~~~~~~


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

my_params={'fit_intercept' : [True, False] }
 
from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(model, my_params, cv=kf, scoring='accuracy',return_train_score=True)

#------------------------------------------------------------------------------
#----------------------->>>> Step 4, Model Fitting  <<<<<----------------------
#------------------------------------------------------------------------------

gs.fit(x,y)


#------------------------------------------------------------------------------
#----------------->>>> Step 5, Getting Info from Model  <<<<<------------------
#------------------------------------------------------------------------------

gs.best_score_  #np.float64(0.947256455399061)

gs.best_params_  #{'fit_intercept': False}

cv_LR=gs.cv_results_ 


new_x = [[14.0, 21.0, 90.0, 590.0, 0.1, 0.08, 0.06, 0.05, 0.19, 0.07, 0.3, 0.9, 2.0, 
                   23.0, 0.006, 0.03, 0.02, 0.02, 0.02, 0.003, 16.0, 30.0, 100.0, 800.0, 
                   0.14, 0.2, 0.15, 0.13, 0.3, 0.09]]

new_x = pd.DataFrame(new_x, columns=data.feature_names)

y_pred = gs.predict(new_x)

print(y_pred)   #[1]






#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


#------------------------------------------------------------------------------
#----------------------->>>> Step 3, Model selection  <<<<<--------------------
#------------------------------------------------------------------------------


#~~~~~~~~~~~~~~~~~~~~~~~>>>>>>> Model: KNN <<<<<<<<~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier()

# LogisticRegression has no hyperparameter

my_params={'n_neighbors': [1,3,5,7,10,15],
           'metric': ['minkowski', 'eucllidiean', 'manhattan']
           }

from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(model, my_params ,cv=kf, scoring='accuracy', return_train_score=True)


#------------------------------------------------------------------------------
#----------------------->>>> Step 4, Model Fitting  <<<<<----------------------
#------------------------------------------------------------------------------

gs.fit(x,y)


#------------------------------------------------------------------------------
#----------------->>>> Step 5, Getting Info from Model  <<<<<------------------
#------------------------------------------------------------------------------

gs.best_params_     #{'metric': 'manhattan', 'n_neighbors': 10}

gs.best_score_    #np.float64(0.9436864241001566)


cv_KNN=gs.cv_results_


new_x = [[14.0, 21.0, 90.0, 590.0, 0.1, 0.08, 0.06, 0.05, 0.19, 0.07, 0.3, 0.9, 2.0, 
                   23.0, 0.006, 0.03, 0.02, 0.02, 0.02, 0.003, 16.0, 30.0, 100.0, 800.0, 
                   0.14, 0.2, 0.15, 0.13, 0.3, 0.09]]

new_x = pd.DataFrame(new_x, columns=data.feature_names)

y_pred = gs.predict(new_x)

print(y_pred)  #[0]



#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


#------------------------------------------------------------------------------
#----------------------->>>> Step 3, Model selection  <<<<<--------------------
#------------------------------------------------------------------------------


#~~~~~~~~~~~~~~~~~~~~~~~>>>>>>> Model: DT  <<<<<<<<~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)

my_params={'max_depth': [1,3,5,7,10,15],
           'min_samples_split':[2,5,10],
           'min_samples_leaf':[2]}


from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model, my_params, cv=kf, scoring='accuracy', return_train_score=True)

#------------------------------------------------------------------------------
#----------------------->>>> Step 4, Model Fitting  <<<<<----------------------
#------------------------------------------------------------------------------

gs.fit(x,y)


#------------------------------------------------------------------------------
#----------------->>>> Step 5, Getting Info from Model  <<<<<------------------
#------------------------------------------------------------------------------

gs.best_params_     #{'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2} 

gs.best_score_     #np.float64(0.9437353286384976)


cv_RF=gs.cv_results_


new_x = [[14.0, 21.0, 90.0, 590.0, 0.1, 0.08, 0.06, 0.05, 0.19, 0.07, 0.3, 0.9, 2.0, 
                   23.0, 0.006, 0.03, 0.02, 0.02, 0.02, 0.003, 16.0, 30.0, 100.0, 800.0, 
                   0.14, 0.2, 0.15, 0.13, 0.3, 0.09]]

new_x = pd.DataFrame(new_x, columns=data.feature_names)

y_pred = gs.predict(new_x)

print(y_pred)  #[1]


#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


#------------------------------------------------------------------------------
#----------------------->>>> Step 3, Model selection  <<<<<--------------------
#------------------------------------------------------------------------------


#~~~~~~~~~~~~~~~~~~~~~~~>>>>>>> Model: RF  <<<<<<<<~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42)

my_params={'n_estimators':[30,40,50,100],
           'max_depth':[10,15,20,25],
           'min_samples_split':[2,5,10],
           'min_samples_leaf':[2],
           'max_features':[5,10]
           }


from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy',return_train_score=True)
#------------------------------------------------------------------------------
#----------------------->>>> Step 4, Model Fitting  <<<<<----------------------
#------------------------------------------------------------------------------


gs.fit(x,y)

#------------------------------------------------------------------------------
#----------------->>>> Step 5, Getting Info from Model  <<<<<------------------
#------------------------------------------------------------------------------

gs.best_params_    #{'max_depth': 10,
#                   'max_features': 5,
 #                   'min_samples_leaf': 2,
 #                   'min_samples_split': 5,
 #                      'n_estimators': 50}

gs.best_score_    #np.float64(0.9701193270735524)


cv_RF=gs.cv_results_


new_x = [[14.0, 21.0, 90.0, 590.0, 0.1, 0.08, 0.06, 0.05, 0.19, 0.07, 0.3, 0.9, 2.0, 
                   23.0, 0.006, 0.03, 0.02, 0.02, 0.02, 0.003, 16.0, 30.0, 100.0, 800.0, 
                   0.14, 0.2, 0.15, 0.13, 0.3, 0.09]]

new_x = pd.DataFrame(new_x, columns=data.feature_names)

y_pred = gs.predict(new_x)

print(y_pred)  #[1]




#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


#------------------------------------------------------------------------------
#----------------------->>>> Step 3, Model selection  <<<<<--------------------
#------------------------------------------------------------------------------


#~~~~~~~~~~~~~~~~~~~~~~~>>>>>>> Model: SVM  <<<<<<<<~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.svm import SVC
model=SVC(random_state=42)


my_params={'kernel':['linear','poly','rbf'],
           'C':[0.01,1],
           'gamma':[0.01,1],
           'degree':[2,3]}

from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy',return_train_score=True)


#------------------------------------------------------------------------------
#----------------------->>>> Step 4, Model Fitting  <<<<<----------------------
#------------------------------------------------------------------------------


gs.fit(x,y)

#------------------------------------------------------------------------------
#----------------->>>> Step 5, Getting Info from Model  <<<<<------------------
#------------------------------------------------------------------------------

gs.best_params_ 

gs.best_score_ 



cv_RF=gs.cv_results_


new_x = [[14.0, 21.0, 90.0, 590.0, 0.1, 0.08, 0.06, 0.05, 0.19, 0.07, 0.3, 0.9, 2.0, 
                   23.0, 0.006, 0.03, 0.02, 0.02, 0.02, 0.003, 16.0, 30.0, 100.0, 800.0, 
                   0.14, 0.2, 0.15, 0.13, 0.3, 0.09]]

new_x = pd.DataFrame(new_x, columns=data.feature_names)

y_pred = gs.predict(new_x)

print(y_pred)  #[1]






