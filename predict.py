import numpy as np
import torch
import matplotlib.pyplot as plt

from parameters import *
import pandas as pd


Validation_data = pd.read_csv('../data/processed/validation_dataset.csv')

design_parameter = Validation_data[featureName]
response = Validation_data[labelNames]

design_parameter = torch.Tensor(design_parameter.values)
######### load model ###################
fwdModel = torch.load("models/fwdModel.pt")
invModel = torch.load("models/invModel.pt")

'''
y_test_pred = fwdModel(design_parameter)
y_test_pred = y_test_pred.detach().numpy()

y_pred = y_test_pred[4,:]

y_test = response.iloc[4,:].to_numpy()

x = range(0, 101)
plt.plot(x, y_pred, '*-', x,y_test)
plt.show()
A = 1
#x_test_pred = invModel(y_test)
'''

fwdTestHistory=pd.read_csv('loss-history/invTestHistory.csv',delimiter=',')
fwdTestHistory= [float(x) for x in fwdTestHistory.columns]
plt.plot(fwdTestHistory)
plt.show()

