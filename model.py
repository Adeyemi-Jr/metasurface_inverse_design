import torch
from parameters import *
import numpy as np
import torch.nn.functional as F


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def createFNN(inputDim,hiddenDim,layers,outputDim):
    model = torch.nn.Sequential()
    model.add_module('Fin',torch.nn.Linear(inputDim, hiddenDim[0]))
    model.add_module('ReluIn',torch.nn.ReLU())
    for i in range(layers):
        model.add_module('F'+str(i+1),torch.nn.Linear(hiddenDim[i],hiddenDim[i+1]))
        model.add_module('Relu'+str(i+1),torch.nn.ReLU())
    model.add_module('Fout',torch.nn.Linear(hiddenDim[layers],outputDim))
    if(randomWeights):
        model.apply(weights_init_uniform_rule)
    return model


def createINN(inputDim,hiddenDim,layers,outputDim):
    model = torch.nn.Sequential()
    model.add_module('Fin',torch.nn.Linear(inputDim, hiddenDim[0]))
    model.add_module('ReluIn',torch.nn.ReLU())
    for i in range(layers):
        model.add_module('F'+str(i+1),torch.nn.Linear(hiddenDim[i],hiddenDim[i+1]))
        model.add_module('Relu'+str(i+1),torch.nn.ReLU())
    model.add_module('Fout',torch.nn.Linear(hiddenDim[layers],outputDim))
    if(randomWeights):
        model.apply(weights_init_uniform_rule)
    return model

def correctionDirect(x):
    data = x.detach().numpy()
    for i in range(data.shape[0]):
        for j in range(1,featureDim):
            if (data[i,j]<thetaMin/2):
                data[i,j]=0
            elif (data[i,j]<thetaMin):
                data[i,j]=thetaMin
    xnew = torch.tensor(data)
    return xnew