import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("kc_house_data.csv")
data.drop('id',axis=1,inplace=True)
data.drop('date',axis=1,inplace=True)

plt.figure(figsize=(15,15))


sns.heatmap(data.corr(),annot=True)
plt.show()


