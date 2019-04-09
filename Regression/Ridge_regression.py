from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


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

R_lr1 = linear_model.Ridge(alpha=1)
R_lr2 = linear_model.Ridge(alpha=100)


R_lr1.fit(X_train,Y_train)
R_lr2.fit(X_train,Y_train)

print("Ridge Regression coefficient for alpha=1")
print("")
print("Intercept", R_lr1.intercept_)
print("coefficient", R_lr1.coef_)


print("Ridge Regression coefficients for alpha=100")
print("")
print("Intercept", R_lr2.intercept_)
print("coefficient", R_lr2.coef_)



pred1 = R_lr1.predict(X_test)
pred2 = R_lr2.predict(X_test)

MSR1 = float(format(np.sqrt(metrics.mean_squared_error(Y_test,pred1)),'.3f'))
print("Mean squared error for alpha=1:",MSR1)
print("="*100)

MSR2 = float(format(np.sqrt(metrics.mean_squared_error(Y_test,pred2)),'.3f'))
print("Mean squared error for alpha=100:",MSR2)
