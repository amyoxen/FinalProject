# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:03:21 2021

@author: akshat
"""
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import *

pd.set_option('display.max_columns', None)

Orig_test_data_A=pd.read_csv('D:/Gtech/Practicum/PROPRIETARY_INFORMATION_Orig_Holdout_Data_Plat_A_Orig_Date.csv')

tmp=[datetime.strptime(i,'%b-%y') for i in Orig_test_data_A['Orig_Date']]
Orig_test_data_A['orig_date']=[i.strftime('%b%Y') for i in tmp]
Orig_test_data_A.drop(['Orig_Date'],axis=1,inplace=True)


Orig_test_data_B=pd.read_csv('D:/Gtech/Practicum/PROPRIETARY_INFORMATION_Orig_Holdout_Data_Plat_B_Orig_Date.csv')
tmp=[datetime.strptime(i,'%Y-%m') for i in Orig_test_data_B['orig_date']]
Orig_test_data_B['orig_date']=[i.strftime('%b%Y') for i in tmp]


Orig_test_data_C=pd.read_csv('D:/Gtech/Practicum/PROPRIETARY_INFORMATION_Orig_Holdout_Data_Plat_C_Orig_Date.csv')

tmp=[datetime.strptime(str(i),'%Y%m') for i in Orig_test_data_C['orig_date']]
Orig_test_data_C['orig_date']=[i.strftime('%b%Y') for i in tmp]

dater=pd.concat([Orig_test_data_A,Orig_test_data_B,Orig_test_data_C])

Orig_test_data=pd.read_excel('D:/Gtech/Practicum/PROPRIETARY_INFORMATION_Orig_Holdout_Data.xlsx',sheet_name='Sheet1')


Orig_test_data=pd.merge(Orig_test_data,dater,on='unique_id')


#store=pd.HDFStore('D:/Gtech/Practicum/notebook/prediction1.h5')


prediction=pd.read_csv('D:/Gtech/Practicum/notebook/prediction (1).csv')



prediction.info()



def PMT(rate, nper,pv, fv=0, type=0):
    if rate!=0:
               pmt = (rate*(fv+pv*(1+ rate)**nper))/((1+rate*type)*(1-(1+ rate)**nper))
    else:
               pmt = (-1*(fv+pv)/nper)  
    return(pmt)


def IPMT(rate, per, nper,pv, fv=0, type=0):
  ipmt = -( ((1+rate)**(per-1)) * (pv*rate + PMT(rate, nper,pv, fv=0, type=0)) - PMT(rate, nper,pv, fv=0, type=0))
  return(ipmt)


def PPMT(rate, per, nper,pv, fv=0, type=0):
  ppmt = PMT(rate, nper,pv, fv=0, type=0) - IPMT(rate, per, nper, pv, fv=0, type=0)
  return(ppmt)

def amortisation_schedule(amount, annualinterestrate, paymentsperyear, years,report_date):

    df = pd.DataFrame({'ppmt' :[PPMT(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(paymentsperyear*years)],
                                 'ipmt' :[IPMT(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(paymentsperyear*years)]})
    df.ppmt=-1*df.ppmt
    df.ipmt=-1*df.ipmt
    df['eom'] = amount - np.cumsum(df.ppmt)
    df['eom'] =df['eom'].round(5)
    df['bom'] = df['eom'] + df.ppmt
    df['mob'] = [ i for i in range(1,years*12+1)]
    use_date = datetime.strptime(report_date, '%b%Y')
    df['report_date'] = [(use_date+relativedelta(months=+i)).strftime('%Y%m') for i in range(1,years*12+1)]
    return(df)
#amortisation_schedule(7500, 0.1075, 12, 3)

dataF=pd.DataFrame()
ls=[]
dataF2=pd.DataFrame()
dataF3=pd.DataFrame()

for index, row in Orig_test_data.iterrows():
    
    f=amortisation_schedule(row['loan_amount'], row['int_rate']/100, 12, int(row['term']/12),row['orig_date'])
    f['unique_id']=row['unique_id']
    #report
    ls.append(f)
    #dataF.append(f,ignore_index=True)
    
dataF=pd.concat(ls)
dataF.to_csv('D:/Gtech/Practicum/notebook/holdout_trans.csv')

dataF
    
#dataF is the test data to be used for enerating probabilities


prediction['key']=prediction['unique_id']+[str(i) for i in prediction['mob']]

dataF['key']=dataF['unique_id']+[str(i) for i in dataF['mob']]


merged_cashflow=pd.merge(dataF,prediction,on='key')
merged_cashflow.info()


prediction.info()
merged_cashflow.head(100)

merged_cashflow['cashflow']= merged_cashflow['bom']*(merged_cashflow['PrePmt_Probability']-merged_cashflow['CO_Probability'])
merged_cashflow.drop(['key','unique_id_y','mob_y'],axis=1,inplace=True)

merged_cashflow.rename(columns={"mob_x": "mob", "unique_id_x": "unique_id"},inplace=True)

merged_cashflow.columns

merged_cashflow=merged_cashflow[['unique_id','ppmt', 'ipmt', 'eom', 'bom', 'mob', 'report_date',
       'CO_Probability',
       'PrePmt_Probability', 'cashflow']]

merged_cashflow.to_csv("D:/Gtech/Practicum/notebook/cashflow.csv")


merged_cashflow.head(10000).to_csv("D:/Gtech/Practicum/notebook/cashflow_sample.csv")

check=merged_cashflow[merged_cashflow['CO_Probability']>=0.5]

check=check[check['PrePmt_Probability']>=0.5]

merged_cashflow.columns

test=merged_cashflow.groupby('unique_id',group_keys=False)[['cashflow','ppmt','ipmt']].agg('sum')


aggregated_cash=pd.merge(Orig_test_data,test,on='unique_id')

aggregated_cash.to_csv('D:/Gtech/Practicum/notebook/aggregated_cashflow.csv')
