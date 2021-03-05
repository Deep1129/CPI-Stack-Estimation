Python code to make linear regression model for any benchmark.
Train and Test data files should be saved as data_train and data_test.


#importing libararies and packages



import pandas as pd
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import warnings
from google.colab import drive
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model





#Read test data and convert it into required CSV file 

read_file = pd.read_fwf (r'data_train.txt')
read_file.to_csv (r'example1.csv', index=None)         #saving CSV file
df1 = pd.read_csv('example1.csv')
df1.drop(df1.columns[[0,1,3,5,6,7,8,9,10]],axis =1)    # dropping extra columns
df1.to_csv('example2.csv')
df2 = pd.read_csv('example2.csv')
df2 = df2.replace(',','', regex=True)
branch_load= []                                        #seperate lists for all events
branch= []                                             
L1D= []
L1I= []
dTLB= []
iTLB= []
cache = []
cycle =[]
instr =[]
for i in df2.index:                                    #filling lists
    if(df2['events'][i] == 'branch-load-misses'):
        branch_load.append(df2['counts'][i])
    elif(df2['events'][i] == 'branch-misses'):
        branch.append(df2['counts'][i])
    elif(df2['events'][i] == 'L1-dcache-load-misses'):
        L1D.append(df2['counts'][i])
    elif(df2['events'][i] == 'L1-icache-load-misses'):
        L1I.append(df2['counts'][i])
    elif(df2['events'][i] == 'dTLB-load-misses'):
        dTLB.append(df2['counts'][i])
    elif(df2['events'][i] == 'iTLB-load-misses'):
        iTLB.append(df2['counts'][i])
    elif(df2['events'][i] == 'cache-misses'):
        cache.append(df2['counts'][i])
    elif(df2['events'][i] == 'instructions'):
        instr.append(df2['counts'][i])
    elif(df2['events'][i] == 'cycles'):
        cycle.append(df2['counts'][i])
cpi = []                                                #CPI as cycles/intructions
for i in range(len(cycle)):
    cpi.append(float(cycle[i])/float(instr[i]))
def nor(l):                                             #Normalizing data
    l = [float(i) for i in l] 
    l = [(float(i)-min(l))/(max(l)-min(l)) for i in l]
    return l
branch_load = nor(branch_load)
cache = nor(cache)
L1D= nor(L1D)
L1I= nor(L1I)
dTLB= nor(dTLB)
branch= nor(branch)
iTLB= nor(iTLB)
#Making dictionary of lists to convert it into dataframe later
dictionary = {'branch-load-misses' : branch_load, 'branch-misses':branch, 'L1-dcache-load-misses':L1D, 'L1-icache-load-misses':L1I, 'dTLB-load-misses':dTLB
      ,'iTLB-load-misses':iTLB, 'cache-misses' :cache, 'CPI' : cpi}
finalDF = pd.DataFrame(dictionary)
finalDF.to_csv('Train_Data.csv')

	



#Reading test data same as above 



read_file1 = pd.read_fwf (r'data_test.txt')
read_file1.to_csv (r'example3.csv', index=None)
tf1 = pd.read_csv('example3.csv')
tf1.drop(tf1.columns[[0,1,3,5,6,7,8,9,10]],axis =1)
tf1.to_csv('example4.csv')
tf2 = pd.read_csv('example4.csv')
tf2 = tf2.replace(',','', regex=True)
branch_load= []
branch= []
L1D= []
L1I= []
dTLB= []
iTLB= []
cache = []
cycle =[]
instr =[]
for i in tf2.index:
    if(tf2['events'][i] == 'branch-load-misses'):
        branch_load.append(tf2['counts'][i])
    elif(tf2['events'][i] == 'branch-misses'):
        branch.append(tf2['counts'][i])
    elif(tf2['events'][i] == 'L1-dcache-load-misses'):
        L1D.append(tf2['counts'][i])
    elif(tf2['events'][i] == 'L1-icache-load-misses'):
        L1I.append(tf2['counts'][i])
    elif(tf2['events'][i] == 'dTLB-load-misses'):
        dTLB.append(tf2['counts'][i])
    elif(tf2['events'][i] == 'iTLB-load-misses'):
        iTLB.append(tf2['counts'][i])
    elif(tf2['events'][i] == 'cache-misses'):
        cache.append(tf2['counts'][i])
    elif(tf2['events'][i] == 'instructions'):
        instr.append(tf2['counts'][i])
    elif(tf2['events'][i] == 'cycles'):
        cycle.append(tf2['counts'][i])

cpi = []
for i in range(len(cycle)):
    cpi.append(float(cycle[i])/float(instr[i]))
def nor(l):
    l = [float(i) for i in l] 
    l = [(float(i)-min(l))/(max(l)-min(l)) for i in l]
    return l
branch_load = nor(branch_load)
cache = nor(cache)
L1D= nor(L1D)
L1I= nor(L1I)
dTLB= nor(dTLB)
branch= nor(branch)
iTLB= nor(iTLB)

dictionary_test = {'branch-load-misses' : branch_load, 'branch-misses':branch, 'L1-dcache-load-misses':L1D, 'L1-icache-load-misses':L1I, 'dTLB-load-misses':dTLB
      ,'iTLB-load-misses':iTLB, 'cache-misses' :cache, 'CPI' : cpi}

test_DATA = pd.DataFrame(dictionary_test)
test_DATA.to_csv('Test_Data.csv')





#Making linear regression model and computing coefficients, intercepts , r squared , adjusted r squared , F stats values using it



X_with_constant = sm.add_constant(finalDF[['branch-load-misses','branch-misses','L1-dcache-load-misses','L1-icache-load-misses','dTLB-load-misses','iTLB-load-misses','cache-misses']])
model = sm.OLS(finalDF.CPI, X_with_constant)
results = model.fit()
results.summary()





#Forcing to be positive

from sklearn.linear_model import Lasso
lin = Lasso(alpha=0,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
lin.fit(finalDF[['branch-load-misses','branch-misses','L1-dcache-load-misses','L1-icache-load-misses','dTLB-load-misses','iTLB-load-misses','cache-misses']],finalDF.CPI)		







#Calculating RMSE value


predictedCPI = lin.predict(test_DATA[['branch-load-misses','branch-misses','L1-dcache-load-misses','L1-icache-load-misses','dTLB-load-misses','iTLB-load-misses','cache-misses']])


actualCPI = test_DATA.CPI
mse = sklearn.metrics.mean_squared_error(actualCPI, predictedCPI)
rmse = math.sqrt(mse)
rmse




#Calculating residual and plotting graph


residual = actualCPI - predictedCPI
sns.distplot(residual)




#Ploting predicted values vs actual values



fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)
