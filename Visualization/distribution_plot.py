import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("kc_house_data.csv")
data.drop('id',axis=1,inplace=True)
data.drop('date',axis=1,inplace=True)

features = ['price','bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']

#color=['b','g','yellow','g','b','purple','cyan','g']

plt.figure(figsize=(20,20))
lst = data[features].columns.values

for i in range(len(features)):
	plt.subplot(4,int(len(lst))//3,i+1)
	sns.kdeplot(data[features[i]],color='purple')
	plt.tight_layout()
plt.show()