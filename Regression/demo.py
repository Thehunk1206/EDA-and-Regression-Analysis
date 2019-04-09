import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("kc_house_data.csv")
data.drop('id',axis=1,inplace=True)
data.drop('date',axis=1,inplace=True)


print(data.head(10),"\n")
print(data.describe(),"\n")
print(data.columns)


