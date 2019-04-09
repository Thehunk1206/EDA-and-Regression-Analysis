import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("kc_house_data.csv")
data.drop('id',axis=1,inplace=True)
data.drop('date',axis=1,inplace=True)



features = ['price','bedrooms', 'bathrooms', 'sqft_living' ,'sqft_lot', 'floors',
			 'condition', 'grade','sqft_living15', 'sqft_lot15']

lst = data[features].columns.values


plt.figure(figsize=(15,15))
sns.set_style('whitegrid')

for i in range(0,len(lst)):
	plt.subplot(1,len(lst),i+1)
	plt.yticks(rotation=90)
	sns.boxplot(data[lst[i]],color='purple',orient='v')
	plt.tight_layout()

plt.show()


