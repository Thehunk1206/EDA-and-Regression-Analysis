from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
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

lr = linear_model.LinearRegression()


lr.fit(X_train,Y_train)

print("Intercept", lr.intercept_)
print("coefficient", lr.coef_)

pred = lr.predict(X_test)


MSR = float(format(np.sqrt(metrics.mean_squared_error(Y_test,pred)),'.3f'))
print("Mean squared error:",MSR)
