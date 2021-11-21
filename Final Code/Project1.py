# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:05:00 2021

@author: akshat
"""


import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)

Plat_A_Orig=pd.read_csv('D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\Zach_Statham_platform_a_orig_data.csv')
Plat_A_Orig.rename(columns={"Orig_Date":"orig_date"},inplace=True)
Plat_A_Orig.columns


Plat_B_Orig=pd.read_csv('D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\Zach_Statham_platform_b_orig_data.csv')
Plat_B_Orig.columns


Plat_C_Orig=pd.read_csv('D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\Zach_Statham_platform_c_orig_data.csv')
Plat_C_Orig.columns
Plat_C_Orig.isna().sum()

'''percent_missing = Plat_C_Orig.isnull().sum() * 100 / len(Plat_C_Orig)
missing_value_df = pd.DataFrame({'column_name': Plat_C_Orig.columns,
                                'percent_missing': percent_missing})
'''

Plat_A_trans=pd.read_csv('D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\Zach_Statham_platform_a_trans_data.csv')
Plat_A_trans.columns

#convert report date to same format as platform B and C --> yearmonth both in numerals ex: 201707 for July 2017 

Plat_A_trans['report_date']=pd.to_datetime(Plat_A_trans['report_date']).dt.strftime('%Y%m')

#Plat_A_trans['report_date']=[pd.to_datetime(i).strftime('%Y%m') for i in Plat_A_trans['report_date']]



Plat_B_trans=pd.read_csv('D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\Zach_Statham_platform_b_trans_data.csv')
Plat_B_trans.columns


Plat_C_trans=pd.read_csv('D:\Gtech\Practicum\OneDrive_2021-09-02\Fall 21 Credigy\Zach_Statham_platform_c_trans_data.csv')
Plat_C_trans.columns

df=Plat_A_Orig.append(Plat_B_Orig,sort=True)
df=df.append(Plat_C_Orig,sort=True)



#df['emi']=-1*np.pmt(df['int_rate']/1200, df['term'], df['loan_amount'])
#df['emi']=[np.pmt(i/1200, j, k) for i,j,k in zip(df['int_rate'],df['term'],df['loan_amount'])]


percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

df2=Plat_A_trans.append(Plat_B_trans,sort=True)
df2=df2.append(Plat_C_trans,sort=True)


ls=[ 'home_ownership', 'fico', 'loan_amount', 'int_rate',
       'term', 'grade', 'monthly_payment', 'income', 'num_tot_accts',
       'num_open_accts', 'inq_last_6m', 'dti', 'accts_opn_last_24m',
       'earliest_cr_line', 'revol_bal', 'util_rate', 'tot_credit_bal',
       'employment_length', 'loan_over_income', 'accts_opn_last_6m',
       'inq_last_12m']

ls1=[ 'home_ownership', 'fico','int_rate','term',]

'''
data_dict={}
for i in ls:
   data_dict[i]=set([x for x in df['{}'.format(i)] if str(x) != 'nan'])
   
df1 = pd.DataFrame(data=data_dict, index=[0])
df1=df1.T
#df1.to_csv('Data_elements.csv')
'''
ls1=['bom', 'co_amt', 'default', 'eom', 'ipmt', 'loan_status', 'mob', 'ppmt',
       'prepay', 'prepay_amt', 'report_date', 'unique_id']


df2.columns
uni_id_trans_group=df2.groupby('unique_id',group_keys=False)

#avg1=uni_id_trans_group[['emi']].agg('mean')
sum1=uni_id_trans_group[['co_amt', 'default','prepay', 'prepay_amt']].agg('sum')
max1=uni_id_trans_group[['mob']].agg('max')


#set(df2['loan_status'])

#uni_id_trans_group=df2.head(1000).groupby('unique_id')

def last(df):
    s=df.tail(1)
    return s.iloc[-1,0]


def mob_diff(df):
    late_16_30_days=df[df['loan_status']=='late_16_30_days']
    if len(late_16_30_days)!=0:
        mob_16=min(late_16_30_days['mob'])
    else:
        mob_16=0
    late_31_120_days=df[df['loan_status']=='late_31_120_days']
    if len(late_31_120_days)!=0:
        mob_31=min(late_31_120_days['mob'])
    else:
        mob_31=0
    return abs(mob_31-mob_16)


mob_first_diff_late_events=uni_id_trans_group.apply(mob_diff)

mob_first_diff_late_events=pd.DataFrame(data=mob_first_diff_late_events,columns=['mob_first_diff_late_events'])

'''
{'charged_off',
 'current',
 'late_16_30_days',
 'late_31_120_days',
 nan,
 'paid_off'}

'''


df2['report_date']
def count_16_30_late(df):
    late_16_30_days=df[df['loan_status']=='late_16_30_days']
    return len(late_16_30_days) 
def count_31_120_late(df):
    late_31_120_days=df[df['loan_status']=='late_31_120_days']
    return len(late_31_120_days)


loan_status=uni_id_trans_group[['loan_status']].apply(last)
count_16_30_late_payment=uni_id_trans_group[['loan_status']].apply(count_16_30_late)
count_31_120_late_payment=uni_id_trans_group[['loan_status']].apply(count_31_120_late)
bom=uni_id_trans_group[['bom']].apply(last)
last_report_date=uni_id_trans_group[['report_date']].apply(last)


loan_status=pd.DataFrame(data=loan_status,columns=['loan_status'])
count_16_30_late_payment=pd.DataFrame(data=count_16_30_late_payment,columns=['count_16_30_late_payment'])
count_31_120_late_payment=pd.DataFrame(data=count_31_120_late_payment,columns=['count_31_120_late_payment'])
bom=pd.DataFrame(data=bom,columns=['bom'])
last_report_date=pd.DataFrame(data=last_report_date,columns=['last_report_date'])


last_report_date['report_month']=[int(str(i)[-2:]) for i in last_report_date['last_report_date']]
last_report_date['report_year']=[ int(str(i)[:4]) for i in last_report_date['last_report_date']]

last_report_date.drop(columns=['last_report_date'],inplace=True)

'''   
df2[df2['unique_id']=='A-100030-204083']
loan_status.index[:][0]
max1.head(4)
'''

frames = [sum1, max1,bom,loan_status,count_16_30_late_payment,count_31_120_late_payment,last_report_date,mob_first_diff_late_events]

summarized_trans=pd.concat(frames,axis=1)


#imputing mssing values for loan_status based on prepay and default indicator
summarized_trans.loc[(summarized_trans.loan_status.isna()) & (summarized_trans.prepay == 1), 'loan_status'] = 'paid_off'
summarized_trans.loc[(summarized_trans.loan_status.isna()) & (summarized_trans.prepay == 0), 'loan_status'] = 'current'
summarized_trans.loc[(summarized_trans.loan_status.isna()) & (summarized_trans.default == 1), 'loan_status'] = 'charged_off'

summarized_trans.to_csv('summarized_trans.csv')


zero_loan=list(df['unique_id'][df['loan_amount']==0])

def first(df):
    s=df.head(1)
    return s.iloc[0,0]

zero_loan_grp=df2[df2['unique_id'].isin(zero_loan)].groupby('unique_id',group_keys=False)

asd=zero_loan_grp[['bom']].apply(first)
#asd[zero_loan]
df.set_index('unique_id',inplace=True)
df.loc[zero_loan,'loan_amount']=[i for i in asd[zero_loan].values]

Merged_data_A_B_C=pd.merge(df,summarized_trans,how='inner',left_on='unique_id',right_on='unique_id',suffixes=('_1','_2'))

Merged_data_A_B_C.to_csv('Merged_data_A_B_C_v1.csv')

