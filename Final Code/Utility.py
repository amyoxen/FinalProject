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


prediction=pd.read_csv('D:/Gtech/Practicum/notebook/prediction_update.csv')



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
    df['loan_amount']=amount
    #df['int_rate']=annualinterestrate
    return(df)
#amortisation_schedule(7500, 0.1075, 12, 3)

dataF=pd.DataFrame()
ls=[]


for index, row in Orig_test_data.iterrows():
    
    f=amortisation_schedule(row['loan_amount'], row['int_rate']/100, 12, int(row['term']/12),row['orig_date'])
    f['unique_id']=row['unique_id']
    #report
    ls.append(f)
    #dataF.append(f,ignore_index=True)
    
dataF=pd.concat(ls)
#dataF.to_csv('D:/Gtech/Practicum/notebook/holdout_trans.csv')
#dataF =pd.read_csv('D:/Gtech/Practicum/notebook/holdout_trans.csv')

   
#dataF is the test data to be used for enerating probabilities

prediction['key']=prediction['unique_id']+[str(i) for i in prediction['mob']]

dataF['key']=dataF['unique_id']+[str(i) for i in dataF['mob']]


merged_cashflow=pd.merge(dataF,prediction,on='key')
#merged_cashflow.info()


merged_cashflow['prepmt']=merged_cashflow['loan_amount']*merged_cashflow['PrePmt_Probability']
merged_cashflow['CO']=merged_cashflow['loan_amount']*merged_cashflow['CO_Probability']

                                                  
merged_cashflow.drop(['key','unique_id_y','mob_y'],axis=1,inplace=True)

merged_cashflow.rename(columns={"mob_x": "mob", "unique_id_x": "unique_id"},inplace=True)

merged_cashflow.columns

merged_cashflow=merged_cashflow[['unique_id','loan_amount','ppmt', 'ipmt', 'eom', 'bom', 'mob', 'report_date','prepmt', 'CO',
       'CO_Probability', 'PrePmt_Probability']]

for i in range(len(merged_cashflow)-1):
        merged_cashflow['eom'].values[i]=merged_cashflow['bom'].values[i]-merged_cashflow['ppmt'].values[i]-merged_cashflow['CO'].values[i]-merged_cashflow['prepmt'].values[i]
        merged_cashflow['bom'].values[i+1]=merged_cashflow['eom'].values[i]


merged_cashflow['cashflow']= merged_cashflow['ppmt']+ merged_cashflow['prepmt']-merged_cashflow['CO']

merged_cashflow.to_csv("D:/Gtech/Practicum/notebook/cashflow_updated_1.csv")

view1=merged_cashflow.groupby('mob',group_keys=False).sum()[['loan_amount','ppmt','ipmt','prepmt','CO','cashflow']]
view1.reset_index(inplace=True)
#view1.to_csv('D:/Gtech/Practicum/notebook/view1.csv')

view2=merged_cashflow.groupby('mob',group_keys=False).mean()[['PrePmt_Probability','CO_Probability']]
view2.reset_index(inplace=True)
combined_view=pd.merge(view1,view2, on='mob')

combined_view['bom']=combined_view['loan_amount']
combined_view['eom']=0
                                   

for i in range(len(combined_view)-1):
        combined_view['eom'].values[i]=combined_view['bom'].values[i]-combined_view['ppmt'].values[i]-combined_view['CO'].values[i]-combined_view['prepmt'].values[i]
        combined_view['bom'].values[i+1]=combined_view['eom'].values[i]

combined_view.to_csv('D:/Gtech/Practicum/notebook/combined_view.csv')


#merged_cashflow=pd.read_csv("D:/Gtech/Practicum/notebook/cashflow_updated_1.csv")
#merged_cashflow.drop(['Unnamed: 0'],axis=1,inplace=True)

