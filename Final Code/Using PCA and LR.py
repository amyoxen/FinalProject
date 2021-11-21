# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:10:00 2021

@author: akshat
"""



import pandas as pd
import numpy as np
import seaborn as sns
from seaborn import boxplot
import matplotlib.pyplot as plt
from matplotlib import pyplot
from seaborn import distplot
import sklearn



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



newdata_regression_imputation=pd.read_csv('D:/Gtech/Practicum/OneDrive_2021-09-02/Fall 21 Credigy/newdata_regression_imputed.csv')



newdata_regression_imputation.columns

'''xval=['accts_opn_last_24m', 'accts_opn_last_6m', 'dti',
       'earliest_cr_line', 'fico', 'home_ownership', 'income', 'inq_last_12m',
       'inq_last_6m', 'int_rate', 'loan_amount', 'loan_over_income',
       'monthly_payment', 'num_open_accts', 'num_tot_accts', 'revol_bal',
       'term', 'tot_credit_bal', 'util_rate', 'co_amt', 'default', 'prepay',
       'prepay_amt', 'mob', 'bom', 'count_16_30_late_payment',
       'count_31_120_late_payment', 'report_month', 'report_year',
       'mob_first_diff_late_events', 'employment_length>=5', 'orig_month',
       'orig_year', 'grade_A', 'grade_AA', 'grade_B', 'grade_C', 'grade_D',
       'grade_E', 'grade_F', 'grade_G', 'grade_HR', 'platform_id_A',
       'platform_id_B', 'platform_id_C', 'loan_status_current',
       'loan_status_late_16_30_days', 'loan_status_late_31_120_days',
       'employment_length', 'PRE_LOAN_DTI', 'POST_LOAN_DTI',
       'Additional_debt_ratio']
'''

xval=['accts_opn_last_24m', 'accts_opn_last_6m', 'dti',
       'earliest_cr_line', 'fico','income', 'inq_last_12m',
       'inq_last_6m', 'int_rate', 'loan_amount', 'loan_over_income',
       'monthly_payment', 'num_open_accts', 'num_tot_accts', 'revol_bal',
       'tot_credit_bal', 'util_rate','mob', 'bom','PRE_LOAN_DTI', 'POST_LOAN_DTI',
       'Additional_debt_ratio']

catval=['home_ownership','term','report_month', 'report_year','employment_length>=5','orig_month',
       'orig_year', 'grade_A', 'grade_AA', 'grade_B', 'grade_C', 'grade_D',
       'grade_E', 'grade_F', 'grade_G', 'grade_HR', 'platform_id_A',
       'platform_id_B', 'platform_id_C']



pd.set_option('display.max_columns', None)
newdata_regression_imputation.describe()


X=newdata_regression_imputation[xval].values


#scaler = sklearn.preprocessing.StandardScaler().fit(X)
from sklearn.preprocessing import QuantileTransformer

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve


qt = QuantileTransformer(n_quantiles=10, random_state=0)
X=qt.fit_transform(X)

X=np.concatenate((X,newdata_regression_imputation[catval].values),axis=1)


#X_prepay=newdata_regression_imputation[xval].values
Y_default=newdata_regression_imputation['default'].values
Y_prepay=newdata_regression_imputation['prepay'].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y_default, test_size=0.2, random_state=0)


clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
Y_pre=clf.predict(X_test)

target_names = ['No Default', 'Default']

print(classification_report(Y_test, Y_pre, target_names=target_names))


#clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
    #logit_clf = LogisticRegression()
    

#clf = LogisticRegression(random_state=0,max_iter=20)
#skf = StratifiedKFold(n_splits=3)
#scores=model_selection.cross_val_score(clf, X,y=Y_default,cv=skf)

newdata_regression_imputation['default'].value_counts()/newdata_regression_imputation['default'].shape[0]
newdata_regression_imputation['prepay'].value_counts()/newdata_regression_imputation['prepay'].shape[0]



dr=[]
prob=[]


w = [{0:5,1:95},{0:3,1:97},{0:10,1:90},{0:12,1:93},{0:15,1:85},{0:20,1:80},{0:25,1:75},{0:30,1:70}]
hyperparam_grid = {"class_weight": w }
clf = LogisticRegression(random_state=7416)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_default, test_size=0.2, random_state=0)

grid = GridSearchCV(clf,hyperparam_grid,scoring="f1", cv=10, n_jobs=-1, refit=True)
grid.fit(X_train,Y_train)
    
print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

grid.cv_results_


pd.DataFrame({'param': grid.cv_results_["params"], 'acc': grid.cv_results_["mean_test_score"]})











w = [{0:10,1:90},{0:11,1:89},{0:12,1:83},{0:13,1:87},{0:14,1:86},{0:15,1:85},{0:16,1:84},{0:17,1:83},{0:18,1:82},{0:19,1:81},{0:20,1:80}]
hyperparam_grid = {"class_weight": w }
clf = LogisticRegression(random_state=7416)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y_default, test_size=0.2, random_state=0)

grid = GridSearchCV(clf,hyperparam_grid,scoring="roc_auc", cv=10, n_jobs=-1, refit=True)
grid.fit(X,Y_default)
    
print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

grid.cv_results_

f1_default=pd.DataFrame({'param': grid.cv_results_["params"], 'acc': grid.cv_results_["mean_test_score"]})






lg3 = LogisticRegression(random_state=13, class_weight={0:17,1:83})
# fit it
lg3.fit(X_train,Y_train)
# test
y_pred = lg3.predict(X_test)
# performance
print(f'Accuracy Score: {accuracy_score(Y_test,y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(Y_test, y_pred)}')
print(f'Area Under Curve: {roc_auc_score(Y_test, y_pred)}')
print(f'Recall score: {recall_score(Y_test,y_pred)}')
target_names = ['No Default', 'Default']
print(classification_report(Y_test, y_pred, target_names=target_names))










######### PREPAY



w = [{0:35,1:65},{0:40,1:60},{0:45,1:55},{0:50,1:50}]
hyperparam_grid = {"class_weight": w }
clf = LogisticRegression(random_state=7416)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y_default, test_size=0.2, random_state=0)

grid = GridSearchCV(clf,hyperparam_grid,scoring="roc_auc", cv=10, n_jobs=-1, refit=True)
grid.fit(X,Y_prepay)
    
print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

grid.cv_results_

f1_prepay=pd.DataFrame({'param': grid.cv_results_["params"], 'acc': grid.cv_results_["mean_test_score"]})









lg3 = LogisticRegression(random_state=13,class_weight={0:45,1:55})
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_prepay, test_size=0.2, random_state=0)


# fit it
lg3.fit(X_train,Y_train)
# test
y_pred = lg3.predict(X_test)
# performance
print(f'Accuracy Score: {accuracy_score(Y_test,y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(Y_test, y_pred)}')
print(f'Area Under Curve: {roc_auc_score(Y_test, y_pred)}')
print(f'Recall score: {recall_score(Y_test,y_pred)}')
target_names = ['No Default', 'Default']
print(classification_report(Y_test, y_pred, target_names=target_names))










