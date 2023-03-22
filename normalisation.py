import numpy
import torch
import pandas
from torch.utils.data import Dataset, TensorDataset, DataLoader


class Normalisation:

    def __init__(self,data):
        self.mu = torch.mean(data,dim=0)
        self.std = torch.std(data, dim=0)
        self.min = torch.min(torch.abs(data), dim=0)[0]
        self.max = torch.max(torch.abs(data), dim=0)[0]
        self.cols = data.size()[1]



    def normalise(self, data):
        for i in range(0,self.cols):
            data[:,i] = torch.div(data[:,i]- self.min[i], self.max[i] -self.min[i])

        return data


    def unnormalise(self, data):
        for i in range(0, self.cols):
            # Scaling based on max value:
            data[:, i] = torch.mul(data[:, i], self.max[i] - self.min[i]) + self.min[i]

        return data

