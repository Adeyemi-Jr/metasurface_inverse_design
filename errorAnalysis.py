import torch
import statistics as stats
from parameters import *

def computeR2(T1,T2):
    #T1: predictions
    #T2: ground truth
    cols = T1.shape[1]
    R2 = torch.zeros(cols)
    for i in range(cols):
        y = T1[:,i:(i+1)]
        x = T2[:,i:(i+1)]
        SSres = torch.norm(x-y)**2
        SStot = torch.norm(x-torch.mean(x))**2
        R2[i] = 1.-SSres/SStot
    return R2