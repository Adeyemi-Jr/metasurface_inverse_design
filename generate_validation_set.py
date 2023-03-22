import numpy
import pandas as pd


df= pd.read_csv('../data/processed/data.csv')

Valfraction = 0.10



# S11
val_data = df.sample(frac=Valfraction)
df.drop(val_data.index,inplace= True)

df.to_csv('../data/processed/data_training.csv')
val_data.to_csv('../data/processed/data_test.csv')


