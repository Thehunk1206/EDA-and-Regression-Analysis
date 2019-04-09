import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("kc_house_data.csv")

data.drop('id',axis=1,inplace=True)
data.drop('date',axis=1,inplace=True)


#fig=plt.figure()

features =  ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']


lst = data[features].columns.values


fig = plt.figure()

for i in range(0,len(lst)):
	ax = fig.add_subplot(3,int(len(lst))/3,i+1)
	ax.scatter(data[features[i]],data['price'],c="darkgreen",alpha=0.5)
	ax.set(xlabel=features[i],ylabel='\nPrice')

plt.show(fig)

