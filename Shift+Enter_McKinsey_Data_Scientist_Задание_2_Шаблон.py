import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# read input
df = pd.read_csv('opsd_austria_daily.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# data imputation
def clean(dataset, columns):
    dataset[columns].replace(0, np.nan, inplace=True)
    dataset[columns].fillna(method=..., inplace=True) # choose a scheme for fillna
# call a clean function here for Electricity_consumption, Wind_production columns

# ensemble learning for regression
def ensemble_training(df_train, df_test):
    label = 'Electricity_consumption'
    X_train = df_train.drop(label, axis=1)
    y_train = df_train.loc[:,label]
    X_test = df_test.drop(label, axis=1)
    y_test = df_test.loc[:,label]
    dtrab = AdaBoostRegressor(DecisionTreeRegressor(max_depth=...),
                              # check which value for max_depth gives better RMSE/MSE: 5, 10, 15, 20, 25
                              n_estimators=100,
                              # check which value for n_estimators gives better RMSE/MAE: 10, 20, 30, 40, 50, 100
                              random_state=1)
    dtrab.fit(X_train, y_train)
    y_predict=dtrab.predict(X_test)
    print('RMSE: %.6f' %(sqrt(mean_squared_error(y_test, y_predict))))
    print('MAE: %.6f' %(mean_absolute_error(y_test, y_predict)))
    df_sol=pd.DataFrame({'True': np.array(y_test),'Predicted': np.array(y_predict)})
    return dtrab, df_sol

features = ['Electricity_consumption','Wind_production','Month']
df_train = df.loc[df['Year']!=2019, features]
df_test = df.loc[df['Year']==2019, features]

def month_select(df, column):
    df_dummy = pd.get_dummies(df[column], prefix='M')
    df_new = pd.concat([df, df_dummy], axis=1)
    df_new = df_new.drop(column, axis=1)
    return df_new

df_train = month_select(df_train, 'Month')
df_test = month_select(df_test, 'Month')

df_sol = []
model, df_sol = ensemble_training(df_train, df_test)
df_sol = pd.concat([df_sol.reset_index(drop=True),
                  pd.Series(df.loc[df['Year']==2019,'Date']).reset_index(drop=True)], axis=1)

# visualization
fig,ax = plt.subplots(figsize=(10,4))
ax.plot_date(df_sol.loc[1:15,'Date'],
             df_sol.loc[1:15,'True'],
             marker='None',
             linestyle = '-',
             color='black', label='True')
ax.plot_date(df_sol.loc[1:15,'Date'],
             df_sol.loc[1:15,'Predicted'],
             marker='o',
             linestyle = '-',
             color='navy', markeredgecolor='navy', label='Predicted')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity consumption')
ax.legend(loc='lower right')
plt.show()
