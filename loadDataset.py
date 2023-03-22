import numpy as np
import torch
import pandas as pd
from parameters import *
from normalisation import Normalisation
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader


def getDataset():

    dir = '../data/processed/'

    data = pd.read_csv(dir+'data_training.csv')



    #-----------------------------#
    #       Initialise Tensors
    #-----------------------------#
    feature_tensor = torch.tensor(data[featureName].values)
    label_tensor = torch.tensor(data[labelNames].values)

    feature_tensor = feature_tensor.double()
    feature_Normalisation = Normalisation(feature_tensor)
    feature_tensor = feature_Normalisation.normalise(feature_tensor)


    dataset = TensorDataset(feature_tensor.float(), label_tensor.float())

    train_split_length = round(len(dataset)*trainSplit)
    test_split_length = round(len(dataset) * testSplit)


    split_ratio = [train_split_length,test_split_length]
    print('train/test/validation: ', split_ratio )
    train_set, test_set = torch.utils.data.random_split(dataset, split_ratio , generator=torch.Generator().manual_seed(42))
    return train_set, test_set, feature_Normalisation




#################################################
def exportTensor(name,data,cols, header=True):
    df=pd.DataFrame.from_records(data.detach().numpy())
    if(header):
        df.columns = cols
    print(name)
    df.to_csv(name+".csv",header=header)

def exportList(name,data):
    arr=np.array(data)
    np.savetxt(name+".csv", [arr], delimiter=',')