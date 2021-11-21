# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:00:36 2021

@author: akshat
"""
#EDA



import pandas as pd
import numpy as np
import seaborn as sns
from seaborn import boxplot
import matplotlib.pyplot as plt
from matplotlib import pyplot
from seaborn import distplot
import sklearn

pd.set_option('display.max_columns', None)

Merged_data_A_B_C=pd.read_csv('D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\Merged_data_A_B_C_v1.csv')
#Merged_data_A_B_C.isnull().sum()

#creating employment length as categorical variable
Merged_data_A_B_C['employment_length>=5'] = [1 if i >=5 else 0 for i in Merged_data_A_B_C['employment_length']]

#Merged_data_A_B_C['inq_last_6m'].replace(-4,np.NaN,inplace=True)
#fig, ax = plt.subplots()
#sns.histplot(Merged_data_A_B_C["dti"], ax=ax)
#ax.set_xlim(1,10)
#ax.set_xticks(range(-1,10))
#plt.show()

#max(Merged_data_A_B_C["dti"])
#sns.boxplot(x=Merged_data_A_B_C["platform_id"], y=Merged_data_A_B_C["dti"])

#data=Merged_data_A_B_C.drop(columns=['unique_id','orig_date']).copy()

#extracting month and year of origination

Merged_data_A_B_C.loc[(Merged_data_A_B_C.platform_id == 'A'), 'orig_date'] = pd.to_datetime(Merged_data_A_B_C.loc[(Merged_data_A_B_C.platform_id == 'A'), 'orig_date']).dt.strftime('%Y%m')
Merged_data_A_B_C['orig_month']=[int(str(i)[-2:]) for i in Merged_data_A_B_C['orig_date']]
Merged_data_A_B_C['orig_year']=[int(str(i)[:4]) for i in Merged_data_A_B_C['orig_date']]

data=Merged_data_A_B_C.drop(columns=['orig_date','employment_length']).copy()
#data=Merged_data_A_B_C.drop(columns=['unique_id','orig_date']).copy()


'''

Column Name	Description		
unique_id	 - unique id to merge loan data		
home_ownership -	home ownership status	MORTGAGE/OWN', 'RENT'	
fico - 	credit score	226 different scores	
loan_amount	 - loan amount		
int_rate - 	interest rate		
term -	length of the loan in months	36, 60	
grade -	loan grade by platform		
monthly_payment	 - monthly loan payment amount		
income - 	annual income		
num_tot_accts	 - number of total credit accounts lifetime		
num_open_accts - 	number of currently open credit accounts		
inq_last_6m	 - credit inquries in the last 6 months		What does -4 means in #credit enquiry, JD: confirmed this was data issue
dti	 - debt to income		
accts_opn_last_24m	 - accounts opened in the last 24 months		
earliest_cr_line -	months since earliest credit account opened		
revol_bal	total  - amount of credit card balance 		
util_rate	 - utiliztion rate of credit card balance		
tot_credit_bal -	total amount of credit balance		
employment_length	- length of employment in years	JD Note: Credigy reccommends converting to categorical, above and below 5 years 	
loan_over_income -	loan amount divided by annual income		
accts_opn_last_6m -	accounts opened in the last 6 months		
platform_id	 - platform key		
inq_last_12m	 - inquires in the last 12 months		
orig_date	loan - originaton month year date		





unique_id	 - unique id to merge loan data	group
loan_status -	monthly status of the loan: current, paid off, delinquent, or charged-off	not equal to "current" or "paid off", make separate column for each
report_date -	month of loan performance	?
mob	 - month on book	max
bom	 - beginning of the month outstanding balance	max
ppmt -	monthly principal payment	get EMI
ipmt -	monthly interest payment	
co_amt	 - charge-off amount	sum
prepay_amt	- prepay amount	sum
eom	 - end of the month outstanding balance	sum
prepay	 - loan prepayment indicator	sum
default -	loan charge-off indicator	sum


'''



# Encoding of Categorical variables

#categorical data

#Note employment_length>=5 is already converted to binary 1 and 0



# Encoding of Categorical variables

#Converting home ownership to category of 1 (MORTGAGE/OWN) and 0(RENT)'''

data['home_ownership'] = [1 if i =='MORTGAGE/OWN' else 0 if i=='RENT' else np.nan for i in data['home_ownership'] ]



col_list=['accts_opn_last_24m', 'accts_opn_last_6m', 'dti',
       'earliest_cr_line', 'fico', 'income', 'inq_last_12m',
       'inq_last_6m', 'int_rate', 'loan_amount', 'loan_over_income',
       'monthly_payment', 'num_open_accts', 'num_tot_accts', 'revol_bal',
       'term', 'tot_credit_bal', 'util_rate', 'emi', 'co_amt', 'default',
       'prepay', 'prepay_amt', 'mob', 'bom', 'count_16_30_late_payment',
       'count_31_120_late_payment', 'report_month', 'report_year',
       'mob_first_diff_late_events', 'employment_length>=5', 'orig_month',
       'orig_year', 'grade_A', 'grade_AA', 'grade_B', 'grade_C', 'grade_D',
       'grade_E', 'grade_F', 'grade_G', 'grade_HR', 'loan_status_current',
       'loan_status_late_16_30_days', 'loan_status_late_31_120_days']




data.isnull().sum()
for i in ['inq_last_6m','accts_opn_last_24m','accts_opn_last_6m','earliest_cr_line','inq_last_12m','num_open_accts','revol_bal','tot_credit_bal','util_rate']:
    data[i]=[np.nan if j<0 else j for j in data[i]]
    
data.isnull().sum()


data_before_imputation=data.copy()

data_before_imputation.to_csv("D:\Gtech\Practicum\data_before_imputation.csv")


###############
###############   LOGISTIC REGRESSION
###############
###############
###############




data = pd.get_dummies(data, columns=["grade"], prefix=["grade"] )

data = pd.get_dummies(data, columns=["platform_id"], prefix=["platform_id"] )
data = pd.get_dummies(data, columns=["loan_status"], prefix=["loan_status"] )
data.drop(columns=['loan_status_charged_off','loan_status_paid_off'],inplace=True)




#handling mising data 
#https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-017-0442-1
#https://www.bmj.com/content/338/bmj.b2393

'''First we need to impute Home ownership. Preffered method is mode for categorical data. '''

data['home_ownership'].fillna(1,inplace=True)

data['home_ownership'].isnull().sum()

#using random value imputation for home ownership

#data3['home_ownership']=[(int(np.random.uniform()*10))%2 if np.isnan(i) else i for i in data3['home_ownership']]

data.columns

data_col=['unique_id', 'accts_opn_last_24m', 'accts_opn_last_6m', 'dti',
       'earliest_cr_line', 'fico', 'home_ownership', 'income', 'inq_last_12m',
       'inq_last_6m', 'int_rate', 'loan_amount', 'loan_over_income',
       'monthly_payment', 'num_open_accts', 'num_tot_accts', 'revol_bal',
       'term', 'tot_credit_bal', 'util_rate', 'emi', 'co_amt', 'default',
       'prepay', 'prepay_amt', 'mob', 'bom', 'count_16_30_late_payment',
       'count_31_120_late_payment', 'report_month', 'report_year',
       'mob_first_diff_late_events', 'employment_length>=5', 'orig_month',
       'orig_year', 'grade_A', 'grade_AA', 'grade_B', 'grade_C', 'grade_D',
       'grade_E', 'grade_F', 'grade_G', 'grade_HR', 'platform_id_A',
       'platform_id_B', 'platform_id_C', 'loan_status_current',
       'loan_status_late_16_30_days', 'loan_status_late_31_120_days']

data['employment_length'] = Merged_data_A_B_C['employment_length'].copy()
#employment length is added back for imputation purpose. Will not be used for classification though aswe created a categorical variable using it.






percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_df = pd.DataFrame({'column_name': data.columns,
                                 'percent_missing': percent_missing})

#creating pre-DTI and post-DTI features using the formula 

data['PRE_LOAN_DTI']=1
data['POST_LOAN_DTI']=1


#data.head(10000).to_csv("test.csv")

#data.drop(columns=['PRE_DTI'],inplace=True)

data['Additional_debt_ratio']=data['monthly_payment']/(data['loan_amount']/12)

data.loc[(data.platform_id_A == 1), 'PRE_LOAN_DTI'] = data.loc[(data.platform_id_A == 1), 'dti']
data.loc[(data.platform_id_B == 1), 'PRE_LOAN_DTI'] = data.loc[(data.platform_id_B == 1), 'dti']
data.loc[(data.platform_id_C == 1), 'PRE_LOAN_DTI'] = data.loc[(data.platform_id_C == 1), 'dti'] - data.loc[(data.platform_id_C == 1), 'Additional_debt_ratio']

data.loc[(data.platform_id_A == 1), 'POST_LOAN_DTI'] = data.loc[(data.platform_id_A == 1), 'dti'] + data.loc[(data.platform_id_A == 1), 'Additional_debt_ratio']
data.loc[(data.platform_id_B == 1), 'POST_LOAN_DTI'] = data.loc[(data.platform_id_B == 1), 'dti'] + data.loc[(data.platform_id_B == 1), 'Additional_debt_ratio']
data.loc[(data.platform_id_C == 1), 'POST_LOAN_DTI'] = data.loc[(data.platform_id_C == 1), 'dti'] 

data.drop(columns=['unique_id'],inplace=True)




### Treating Outliers

data_removing_missing=data.copy()
data_removing_missing.dropna(inplace=True)



# IQR
Q1 = np.percentile(data_removing_missing['accts_opn_last_6m'], 25,
                   interpolation = 'midpoint')
 
Q3 = np.percentile(data_removing_missing['accts_opn_last_6m'], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1

#Analyzing  
p=['accts_opn_last_24m', 'accts_opn_last_6m', 'dti',
       'earliest_cr_line', 'fico', 'home_ownership', 'income', 'inq_last_12m',
       'inq_last_6m', 'int_rate', 'loan_over_income',
       'monthly_payment', 'num_open_accts', 'num_tot_accts','term', 'util_rate']

#p=['tot_credit_bal']


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
fig, ax123=plt.subplots(nrows=4,ncols=4,figsize=(20,10),dpi=200)
#plt.autoscale()
k=0
l=0
for i in p:
    if l-4==0:
        l=0
        k=k+1
      
    ax=ax123[l,k]
    
    df1 = data_removing_missing[i].value_counts().reset_index().sort_values('index').reset_index(drop=True)
    df1.columns = [i, 'Frequency']
    s=df1['Frequency'].sum()
    if i=='accts_opn_last_6m':
        df1['Frequency']=(df1['Frequency']*100)/s
        df1.sort_values(by=[i],inplace=True,ascending=True)
        ax.bar(df1[i].head(50),df1['Frequency'].head(50))
        ax.set_xlabel(i)
        ax.set_ylabel("% of total occurences")

    elif i=='monthy_payment':
        ax.bar(df1[i],df1['Frequency'])
        ax.set_xlabel(i)
        ax.set_ylabel("Frequency")     
    elif i=='income':
        ax.bar(df1[i],df1['Frequency'])
        ax.set_xlabel(i)
        ax.set_ylabel("Frequency")     
               
    elif i=='dti':
        ax.bar(df1[i],df1['Frequency'])
        ax.set_xlabel(i)
        ax.set_ylabel("Frequency")

    elif i=='fico':
        ax.bar(df1[i],df1['Frequency'])
        ax.set_xlabel(i)
        ax.set_ylabel("Frequency")
        
    else:
        df1['Frequency']=(df1['Frequency']*100)/s
        df1.sort_values(by=['Frequency'],inplace=True,ascending=False)
        ax.bar(df1[i].head(50),df1['Frequency'].head(50))
        ax.set_xlabel(i)
        ax.set_ylabel("% of total occurences")
    #print(df1.head(20))
    
    #
    #plt.show()
    #plt.close()
    l=l+1
plt.tight_layout()
plt.savefig("Distribution plot using %")
plt.show()
plt.close()
        
    
    
   
data.to_csv("D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\data.csv")


data.describe()

#data.head(3)

#using bayesianridge estimator

'''from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(Merged_data_A_B_C)
'''

data.columns
imp_col=['accts_opn_last_24m', 'accts_opn_last_6m', 'dti',
       'earliest_cr_line', 'fico', 'home_ownership', 'income', 'inq_last_12m',
       'inq_last_6m', 'int_rate', 'loan_amount', 'loan_over_income',
       'monthly_payment', 'num_open_accts', 'num_tot_accts', 'revol_bal',
       'term', 'tot_credit_bal', 'util_rate', 'employment_length','grade_A', 'grade_AA', 'grade_B', 'grade_C', 'grade_D',
       'grade_E', 'grade_F', 'grade_G', 'grade_HR', 'platform_id_A',
       'platform_id_B', 'platform_id_C','PRE_LOAN_DTI', 'POST_LOAN_DTI','Additional_debt_ratio']

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#count = np.isinf(data_removing_missing).values.sum()
#data.columns.to_series()[np.isinf(data_removing_missing).any()]

from sklearn.linear_model import BayesianRidge
imp_mean = IterativeImputer(estimator=(BayesianRidge()),random_state=0,sample_posterior=True)
newdata_regression_imputation=pd.DataFrame(data=imp_mean.fit_transform(data),columns=data.columns)


'''from sklearn.tree import DecisionTreeRegressor
imp_mean = IterativeImputer(estimator=(DecisionTreeRegressor()),random_state=0)
newdata_tree_imputation=pd.DataFrame(data=imp_mean.fit_transform(data3),columns=data3.columns)
'''

newdata_regression_imputation.isnull().sum()

newdata_regression_imputation.to_csv("D:/Gtech/Practicum/OneDrive_2021-09-02/Fall 21 Credigy/newdata_regression_imputed.csv")

newdata_regression_imputation.describe()

newdata_regression_imputation.head(5)
data[data['accts_opn_last_24m'].isnull()]

data.loc[398287]
newdata_regression_imputation.loc[398287]



#EDA - Correlation matrix and Boxplots

data=pd.read_csv('D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\data.csv')

newdata_regression_imputation=pd.read_csv('D:/Gtech/Practicum/OneDrive_2021-09-02/Fall 21 Credigy/newdata_regression_imputed.csv')

#newdata_regression_imputation=pd.read_csv("newdata_regression_imputed.csv")

data_removing_missing=data.copy()
data_removing_missing.dropna(inplace=True)


sns.set(style="whitegrid")
#data = sns.load_dataset("Merged_data_A_B_C")


colnames=['accts_opn_last_24m', 'accts_opn_last_6m', 'dti',
       'earliest_cr_line', 'employment_length', 'fico', 'grade',
       'home_ownership', 'income', 'inq_last_12m', 'inq_last_6m', 'int_rate',
       'loan_amount', 'loan_over_income', 'monthly_payment', 'num_open_accts',
       'num_tot_accts', 'platform_id', 'revol_bal',
       'term', 'tot_credit_bal','util_rate', 'emi', 'co_amt', 'prepay', 'prepay_amt', 'bom', 'mob']

## Removed Unique ID , orig_date 



#fig, axs = plt.subplots(ncols=len(colnames))
#for i in range(len(colnames)):
#    sns.boxplot(x="default", y=colnames[i], data=Merged_data_A_B_C)
#    plt.show()


df=data[['accts_opn_last_24m', 'accts_opn_last_6m','earliest_cr_line',
       'fico', 'home_ownership', 'income', 'inq_last_12m', 'inq_last_6m',
       'int_rate', 'loan_amount', 'loan_over_income', 'monthly_payment',
       'num_open_accts', 'num_tot_accts', 'revol_bal', 'term',
       'tot_credit_bal', 'util_rate','co_amt', 'default', 'prepay',
       'prepay_amt', 'bom', 'mob','PRE_LOAN_DTI', 'POST_LOAN_DTI',
       'Additional_debt_ratio','report_month','orig_month']]


df_r=data_removing_missing[['accts_opn_last_24m', 'accts_opn_last_6m','earliest_cr_line',
       'fico', 'home_ownership', 'income', 'inq_last_12m', 'inq_last_6m',
       'int_rate', 'loan_amount', 'loan_over_income', 'monthly_payment',
       'num_open_accts', 'num_tot_accts', 'revol_bal', 'term',
       'tot_credit_bal', 'util_rate','co_amt', 'default', 'prepay',
       'prepay_amt', 'bom', 'mob','PRE_LOAN_DTI', 'POST_LOAN_DTI',
       'Additional_debt_ratio','report_month','orig_month']]

data_removing_missing.columns


from matplotlib.pyplot import figure

#CORRELATION MATRIX

figure(figsize=(25, 25), dpi=100)
sns.set(font_scale=2)
sns.heatmap(df.corr(), annot=True,linewidths=.5, cmap="YlGnBu",annot_kws={"size": 5},vmin=-1, vmax=1)
plt.savefig("Correlation plot.png",dpi=100)
plt.show()
plt.close()


#Comparing distribution of data pre and post imputation
percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_df = pd.DataFrame({'column_name': data.columns,
                                 'percent_missing': percent_missing})


imputed_columns=missing_value_df['column_name'][missing_value_df['percent_missing']!=0].values

#max(data_removing_missing['accts_opn_last_6m'])

sns.set(font_scale=1)


'''fig, ax123=plt.subplots(nrows=len(imputed_columns),ncols=2,figsize=(15,20),dpi=200)

i=0;
j=0;

for name in imputed_columns:
    print(name)
    sns.histplot(newdata_regression_imputation,x=name,ax=ax123[i, 0],kde=True)
    sns.histplot(data_removing_missing,x=name,ax=ax123[i, 1],kde=True)

    i=i+1
    
plt.tight_layout()
plt.show()
plt.close()'''




names1=['accts_opn_last_24m', 'accts_opn_last_6m', 'earliest_cr_line',
       'fico', 'monthly_payment' ]

names2=['inq_last_12m','inq_last_6m','int_rate', 'loan_over_income','num_open_accts']

names3=['num_tot_accts', 'revol_bal', 'term','tot_credit_bal', 'util_rate']

names4=['mob','PRE_LOAN_DTI', 'POST_LOAN_DTI','income','bom']

sns.__version__

names_1=[names1,names2]
names_2=[names3,names4]


for k in names_1:
    sns.set(font_scale=2)
    fig, ax123=plt.subplots(nrows=2,ncols=5,figsize=(15,20),dpi=200)
    
    i=0;
    j=0;
    
    for name in k:
        print(name)
        sns.boxplot(x=data["default"], y=data[name], ax=ax123[0,i])
        sns.boxplot(x=data["prepay"], y=data[name], ax=ax123[1, i])
        i=i+1
        
        
    plt.tight_layout()
    plt.savefig("BOX plot 1 {}.png".format(k),dpi=150)
    plt.show()
    plt.close()


for k in names_2:
    sns.set(font_scale=2)
    fig, ax123=plt.subplots(nrows=2,ncols=5,figsize=(15,20),dpi=200)
    
    i=0;
    j=0;
    
    for name in k:
        print(name)
        sns.boxplot(x=data["default"], y=data[name], ax=ax123[0, i])
        sns.boxplot(x=data["prepay"], y=data[name], ax=ax123[1, i])
        i=i+1
        
        
    plt.tight_layout()
    plt.savefig("BOX plot 2 {}.png".format(k),dpi=150)
    plt.show()
    plt.close()


#sns.boxplot(x="default", y="mob", data=Merged_data_A_B_C)

# Conclusion

newdata_regression_imputation[newdata_regression_imputation['default']==1].count()

newdata_regression_imputation.head(1)


fig, ax = plt.subplots()
sns.histplot(newdata_regression_imputation["dti"], ax=ax)
ax.set_xlim(1,31)
ax.set_xticks(range(-1,1))
plt.show()

data.info()


