# %%
import pandas as pd,numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import datasets
from sklearn import svm
from tabulate import tabulate

# %%
data=pd.read_csv("CCPP_data.csv")

# %%
df=pd.DataFrame(data)
df.info()

# %%
df.describe()

# %%
df.corr()

# %%
data.plot.scatter(x='AT', y='PE')

# %%
data.plot.scatter(x='PE', y='AT')

# %%
data.plot.scatter(x='V', y='PE')

# %%
data.plot.hexbin(x='AT', y='PE', gridsize=15)

# %%
data.plot.hexbin(x='V', y='PE', gridsize=15)

# %%
data.plot.line()

# %%
df_case1=df[['AT']]

# %%
df_case2=df[['AT','V']]

# %%
df_case3=df[['AT','V','RH']]

# %%
df_case4=df[['AT','V','RH','AP']]

# %%
target_case=df['PE']

# %%
X_train, X_test, y_train, y_test = train_test_split(df_case1, target_case, test_size = 0.2, random_state = 0)

# %%
X_train

# %%
X_train.shape,y_train.shape

# %%
X_test.shape,y_test

# %%
def prediction(y_test,y_prediction):
    mean_sq_error=mean_squared_error(y_test,y_prediction)
    print("mean_sq_error="+mean_sq_error)
    root_mean_sq_error = np.sqrt(mean_sq_error)
    print("root_mean_sq_error="+root_mean_sq_error)
    r_sq = r2_score(y_test, y_prediction)
    print("r_sq="+r_sq)
    mean_abs_error = mean_absolute_error(y_test, y_prediction)
    print("mean_abs_error="+mean_abs_error)

# %%
lt_regression = LinearRegression()
lt_regression.fit(X_train, y_train)

# %%
y_prediction = lt_regression.predict(X_test)
y_prediction

# %%
m1_lt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m1_lt_mean_sq_error

# %%
m1_lt_root_mean_sq_error = np.sqrt(m1_lt_mean_sq_error)
m1_lt_root_mean_sq_error

# %%
m1_lt_r_sq = r2_score(y_test, y_prediction)
m1_lt_r_sq

# %%
m1_lt_mean_abs_error = mean_absolute_error(y_test, y_prediction)
m1_lt_mean_abs_error

# %%
dt_regression= DecisionTreeRegressor()
dt_regression.fit(X_train, y_train)

# %%
y_prediction = dt_regression.predict(X_test)
y_prediction

# %%
m1_dt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m1_dt_mean_sq_error

# %%
m1_dt_root_mean_sq_error = np.sqrt(m1_dt_mean_sq_error)
m1_dt_root_mean_sq_error

# %%
m1_dt_r_sq = r2_score(y_test, y_prediction)
m1_dt_r_sq

# %%
m1_dt_mean_abs_error = mean_absolute_error(y_test, y_prediction)
m1_dt_mean_abs_error

# %%
rf_regression= RandomForestRegressor()
rf_regression.fit(X_train, y_train)

# %%
y_prediction = rf_regression.predict(X_test)
y_prediction

# %%
m1_rt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m1_rt_mean_sq_error

# %%
m1_rt_root_mean_sq_error = np.sqrt(m1_rt_mean_sq_error)
m1_rt_root_mean_sq_error

# %%
m1_rt_r_sq = r2_score(y_test, y_prediction)
m1_rt_r_sq

# %%
m1_rt_mean_abs_error = mean_absolute_error(y_test, y_prediction)
m1_rt_mean_abs_error

# %%
X_train, X_test, y_train, y_test = train_test_split(df_case2, target_case, test_size = 0.2, random_state = 0)

# %%
lt_regression = LinearRegression()
lt_regression.fit(X_train, y_train)

# %%
y_prediction = lt_regression.predict(X_test)
y_prediction

# %%
m2_lt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m2_lt_mean_sq_error

# %%
m2_lt_root_mean_sq_error = np.sqrt(m2_lt_mean_sq_error)
m2_lt_root_mean_sq_error

# %%
m2_lt_r_sq = r2_score(y_test, y_prediction)
m2_lt_r_sq

# %%
m2_lt_mean_abs_error = mean_absolute_error(y_test, y_prediction)
m2_lt_mean_abs_error

# %%
dt_regression= DecisionTreeRegressor()
dt_regression.fit(X_train, y_train)
y_prediction = dt_regression.predict(X_test)
y_prediction

# %%
m2_dt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m2_dt_root_mean_sq_error = np.sqrt(m2_dt_mean_sq_error)
m2_dt_r_sq = r2_score(y_test, y_prediction)
m2_dt_mean_abs_error = mean_absolute_error(y_test, y_prediction)

# %%
m2_dt_mean_sq_error

# %%
m2_dt_root_mean_sq_error

# %%
m2_dt_r_sq

# %%
m2_dt_mean_abs_error

# %%
rf_regression= RandomForestRegressor()
rf_regression.fit(X_train, y_train)
y_prediction = rf_regression.predict(X_test)
y_prediction

# %%
m2_rt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m2_rt_root_mean_sq_error = np.sqrt(m2_rt_mean_sq_error)
m2_rt_r_sq = r2_score(y_test, y_prediction)
m2_rt_mean_abs_error = mean_absolute_error(y_test, y_prediction)

# %%
m2_rt_mean_sq_error

# %%
m2_rt_root_mean_sq_error

# %%
m2_rt_r_sq

# %%
m2_rt_mean_abs_error

# %%
X_train, X_test, y_train, y_test = train_test_split(df_case3, target_case, test_size = 0.2, random_state = 0)

# %%
lt_regression = LinearRegression()
lt_regression.fit(X_train, y_train)
y_prediction = lt_regression.predict(X_test)
y_prediction

# %%
m3_lt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m3_lt_root_mean_sq_error = np.sqrt(m3_lt_mean_sq_error)
m3_lt_r_sq = r2_score(y_test, y_prediction)
m3_lt_mean_abs_error = mean_absolute_error(y_test, y_prediction)

# %%
m3_lt_mean_sq_error

# %%
m3_lt_root_mean_sq_error

# %%
m3_lt_r_sq

# %%
m3_lt_mean_abs_error

# %%
dt_regression= DecisionTreeRegressor()
dt_regression.fit(X_train, y_train)
y_prediction = dt_regression.predict(X_test)
y_prediction

# %%
m3_dt_mean_sq_error=mean_squared_error(y_test,y_prediction)

m3_dt_root_mean_sq_error = np.sqrt(m3_dt_mean_sq_error)

m3_dt_r_sq = r2_score(y_test, y_prediction)

m3_dt_mean_abs_error = mean_absolute_error(y_test, y_prediction)


# %%
m3_dt_mean_sq_error

# %%
m3_dt_root_mean_sq_error

# %%
m3_dt_r_sq

# %%
m3_dt_mean_abs_error

# %%
rf_regression= RandomForestRegressor()
rf_regression.fit(X_train, y_train)
y_prediction = rf_regression.predict(X_test)
y_prediction

# %%
m3_rt_mean_sq_error=mean_squared_error(y_test,y_prediction)

m3_rt_root_mean_sq_error = np.sqrt(m3_rt_mean_sq_error)

m3_rt_r_sq = r2_score(y_test, y_prediction)

m3_rt_mean_abs_error = mean_absolute_error(y_test, y_prediction)


# %%
m3_rt_mean_sq_error

# %%
m3_rt_root_mean_sq_error

# %%
m3_rt_r_sq

# %%
m3_rt_mean_abs_error

# %%
X_train, X_test, y_train, y_test = train_test_split(df_case4, target_case, test_size = 0.2, random_state = 0)

# %%
lt_regression = LinearRegression()
lt_regression.fit(X_train, y_train)


# %%
y_prediction = lt_regression.predict(X_test)
y_prediction

# %%
m4_lt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m4_lt_mean_sq_error

# %%
m4_lt_root_mean_sq_error = np.sqrt(m4_lt_mean_sq_error)
m4_lt_root_mean_sq_error

# %%
m4_lt_r_sq = r2_score(y_test, y_prediction)
m4_lt_r_sq

# %%
m4_lt_mean_abs_error = mean_absolute_error(y_test, y_prediction)
m4_lt_mean_abs_error

# %%
dt_regression= DecisionTreeRegressor()
dt_regression.fit(X_train, y_train)

# %%
y_prediction = dt_regression.predict(X_test)
y_prediction

# %%
m4_dt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m4_dt_mean_sq_error

# %%
m4_dt_root_mean_sq_error = np.sqrt(m4_dt_mean_sq_error)
m4_dt_root_mean_sq_error

# %%
m4_dt_r_sq = r2_score(y_test, y_prediction)
m4_dt_r_sq

# %%
m4_dt_mean_abs_error = mean_absolute_error(y_test, y_prediction)
m4_dt_mean_abs_error

# %%
rf_regression= RandomForestRegressor()
rf_regression.fit(X_train, y_train)

# %%
y_prediction = rf_regression.predict(X_test)
y_prediction

# %%
m4_rt_mean_sq_error=mean_squared_error(y_test,y_prediction)
m4_rt_mean_sq_error

# %%
m4_rt_root_mean_sq_error = np.sqrt(m4_rt_mean_sq_error)
m4_rt_root_mean_sq_error

# %%
m4_rt_r_sq = r2_score(y_test, y_prediction)
m4_rt_r_sq

# %%
m4_rt_mean_abs_error = mean_absolute_error(y_test, y_prediction)
m4_rt_mean_abs_error

# %%
lt_data_head=[["MSE",m1_lt_mean_sq_error,m2_lt_mean_sq_error,m3_lt_mean_sq_error,m4_lt_mean_sq_error],["RMSE",m1_lt_root_mean_sq_error,m2_lt_root_mean_sq_error,m3_lt_root_mean_sq_error,m4_lt_root_mean_sq_error],["RSQ",m1_lt_r_sq,m2_lt_r_sq,m3_lt_r_sq,m4_lt_r_sq],["MABSE",m1_lt_mean_abs_error,m2_lt_mean_abs_error,m3_lt_mean_abs_error,m4_lt_mean_abs_error]]
# m1_lt_data_val=[m1_lt_mean_sq_error,m1_lt_root_mean_sq_error,m1_lt_r_sq,m1_lt_mean_abs_error]

lt_data_head

# %%
col_names = ["LinearRegression","M1", "M2","M3","M4"]
print(tabulate(lt_data_head,headers=col_names))

# %%
dt_data_head=[["MSE",m1_dt_mean_sq_error,m2_dt_mean_sq_error,m3_dt_mean_sq_error,m4_dt_mean_sq_error],["RMSE",m1_dt_root_mean_sq_error,m2_dt_root_mean_sq_error,m3_dt_root_mean_sq_error,m4_dt_root_mean_sq_error],["RSQ",m1_dt_r_sq,m2_dt_r_sq,m3_dt_r_sq,m4_dt_r_sq],["MABSE",m1_dt_mean_abs_error,m2_dt_mean_abs_error,m3_dt_mean_abs_error,m4_dt_mean_abs_error]]
# m1_lt_data_val=[m1_lt_mean_sq_error,m1_lt_root_mean_sq_error,m1_lt_r_sq,m1_lt_mean_abs_error]

dt_data_head

# %%
col_names = ["DecisionTreeRegression","M1", "M2","M3","M4"]
print(tabulate(dt_data_head,headers=col_names))

# %%
rt_data_head=[["MSE",m1_rt_mean_sq_error,m2_rt_mean_sq_error,m3_rt_mean_sq_error,m4_rt_mean_sq_error],["RMSE",m1_rt_root_mean_sq_error,m2_rt_root_mean_sq_error,m3_rt_root_mean_sq_error,m4_rt_root_mean_sq_error],["RSQ",m1_rt_r_sq,m2_rt_r_sq,m3_rt_r_sq,m4_rt_r_sq],["MABSE",m1_rt_mean_abs_error,m2_rt_mean_abs_error,m3_rt_mean_abs_error,m4_rt_mean_abs_error]]
# m1_lt_data_val=[m1_lt_mean_sq_error,m1_lt_root_mean_sq_error,m1_lt_r_sq,m1_lt_mean_abs_error]

rt_data_head

# %%
col_names = ["RandomForestRegression","M1", "M2","M3","M4"]
print(tabulate(rt_data_head,headers=col_names))


