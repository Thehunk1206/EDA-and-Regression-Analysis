from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


#Reading and spliting data into test and train
df = pd.read_csv("kc_house_data.csv")
train_data,test_data = train_test_split(df,train_size=0.8,random_state=3)

features = [ 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15' ]
X_train = np.array(train_data[features])
Y_train = np.array(train_data['price'])

X_test = np.array(test_data[features])
Y_test = np.array(test_data['price'])


#===============================Linear Models============================
#Simple multiple linear regression
lr = linear_model.LinearRegression()

#Ridge Regression
R_lr1 = linear_model.Ridge(alpha=1)
R_lr2 = linear_model.Ridge(alpha=10)

#Lasso Regression
L_r1 = linear_model.Lasso(alpha=1)
L_r2 = linear_model.Lasso(alpha=10)


#Polynomial Regression
polyfeat = PolynomialFeatures(degree=2)
X_all_feat_poly = polyfeat.fit_transform(df[features])
X_trainpoly = polyfeat.fit_transform(train_data[features])
X_testpoly = polyfeat.fit_transform(test_data[features])


#=========================Fitting==============================
lr.fit(X_train,Y_train)

R_lr1.fit(X_train,Y_train)
R_lr2.fit(X_train,Y_train)

L_r1.fit(X_train,Y_train)
L_r2.fit(X_train,Y_train)

poly = linear_model.LinearRegression().fit(X_trainpoly, train_data['price'])

#======================prediction==================================
lr_pred = lr.predict(X_test)

R_lr1_pred = R_lr1.predict(X_test)
R_lr2_pred = R_lr2.predict(X_test)

L_r1_pred = L_r1.predict(X_test)
L_r2_pred = L_r2.predict(X_test)

Poly_pred = poly.predict(X_testpoly)

#============================Model acuaracy============================
#Multiple regression
print("Multiple regression")
lr_MSR = float(format(np.sqrt(metrics.mean_squared_error(Y_test,lr_pred)),'.3f'))
print("Mean squared error:",lr_MSR)
print("")

#Ridge regression
print("Ridge regression")
R_lr1_MSR = float(format(np.sqrt(metrics.mean_squared_error(Y_test,R_lr1_pred)),'.3f'))
print("Mean squared error for alpha=1:",R_lr1_MSR)

R_lr2_MSR = float(format(np.sqrt(metrics.mean_squared_error(Y_test,R_lr2_pred)),'.3f'))
print("Mean squared error for alpha=100:",R_lr2_MSR)
print("")

#Lasso regressio
print("Lasso regression")
L_r1_MSR = float(format(np.sqrt(metrics.mean_squared_error(Y_test,L_r1_pred)),'.3f'))
print("Mean squared error for alpha=1:",L_r1_MSR)

L_r2_MSR = float(format(np.sqrt(metrics.mean_squared_error(Y_test,L_r2_pred)),'.3f'))
print("Mean squared error for alpha=100:",L_r2_MSR)
print("")

#Polynomial regression
print("Polynomial regression")
Poly_MSR = float(format(np.sqrt(metrics.mean_squared_error(Y_test,Poly_pred)),'.3f'))
print("Mean squared error for degree=2:",Poly_MSR)
