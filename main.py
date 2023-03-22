# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import pandas as pd
from parameters import *
from normalisation import Normalisation
from loadDataset import *
from model import *
from errorAnalysis import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    torch.manual_seed(0)
    os.system('mkdir models')
    os.system('mkdir loss-history')

    ##############---FWD MODEL---##############
    fwdModel = createFNN(featureDim, fwdHiddenDim, fwdHiddenLayers, labelDim)
    fwdOptimizer = torch.optim.Adam(fwdModel.parameters(), lr=fwdLearningRate)
    print('\n\n**************************************************************')
    print('fwdModel', fwdModel)
    print('**************************************************************\n')

    ##############---INV MODEL---##############
    invModel = createINN(labelDim, invHiddenDim, invHiddenLayers, featureDim)
    invOptimizer = torch.optim.Adam(invModel.parameters(), lr=invLearningRate)
    print('\n\n**************************************************************')
    print('invModel', invModel)
    print('**************************************************************\n')

    ##############---INIT DATA---##############
    train_set, test_set, featureNormalization = getDataset()
    train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=batchSize,
                                   shuffle=batchShuffle)
    test_data_loader = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set), shuffle=False)

    ##############---Training---##############
    fwdEpochLoss = 0.0
    invEpochLoss = 0.0

    fwdTrainHistory = []
    fwdTestHistory = []
    invTrainHistory = []
    invTestHistory = []
    loader_all_train = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=len(train_set), shuffle=False)
    loader_all_test = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set), shuffle=False)
    x_all_train, y_all_train = next(iter(loader_all_train))
    x_all_test, y_all_test = next(iter(loader_all_test))

    if (fwdTrain):
        print('\nBeginning forward model training')
        print('-------------------------------------')
        ##############---FWD TRAINING---##############
        for fwdEpochIter in range(fwdEpochs):
            fwdEpochLoss = 0.0
            for iteration, batch in enumerate(train_data_loader, 0):
                # get batch
                x_train = batch[0]
                y_train = batch[1]
                # set train mode
                fwdModel.train()
                # predict
                y_train_pred = fwdModel(x_train)
                # compute loss
                fwdLoss = fwdLossFn(y_train_pred, y_train)
                # optimize
                fwdOptimizer.zero_grad()
                fwdLoss.backward()
                fwdOptimizer.step()
                # store loss
                fwdEpochLoss += fwdLoss.item()
            print(" {}:{}/{} | fwdEpochLoss: {:.2e} | invEpochLoss: {:.2e}".format( \
                "fwd", fwdEpochIter, fwdEpochs, fwdEpochLoss / len(train_data_loader),
                                                invEpochLoss / len(train_data_loader)))
            fwdTrainHistory.append(fwdLossFn(fwdModel(x_all_train), y_all_train).item())
            fwdTestHistory.append(fwdLossFn(fwdModel(x_all_test), y_all_test).item())
        print('-------------------------------------')
        # save model
        torch.save(fwdModel, "models/fwdModel.pt")
        # export loss history
        exportList('loss-history/fwdTrainHistory', fwdTrainHistory)
        exportList('loss-history/fwdTestHistory', fwdTestHistory)
    else:
        fwdModel = torch.load("models/fwdModel.pt")
        fwdModel.eval()

    if (invTrain):
        print('\nBeginning inverse model training')
        print('-------------------------------------')
        ##############---INV TRAINING---##############
        for invEpochIter in range(invEpochs):
            invEpochLoss = 0.0

            # Scheduling betaX:
            if (invEpochIter < betaXEpochSchedule):
                betaVal = betaX
            else:
                betaVal = 0

            for iteration, batch in enumerate(train_data_loader, 0):
                # get batch
                x_train = batch[0]
                y_train = batch[1]
                # set train mode
                invModel.train()
                # predict
                x_train_pred = invModel(y_train)
                y_train_pred_pred = fwdModel(x_train_pred)
                # compute loss
                invLoss = invLossFn(y_train_pred_pred, y_train) + betaVal * invLossFn(x_train_pred, x_train)
                # optimize
                invOptimizer.zero_grad()
                invLoss.backward()
                invOptimizer.step()
                # store loss
                invEpochLoss += invLoss.item()
            print(" {}:{}/{} | betaX: {:.2e} | fwd EpochLoss: {:.2e} | invEpochLoss: {:.6e}".format( \
                "inv", invEpochIter, invEpochs, betaVal, fwdEpochLoss / len(train_data_loader),
                                                         invEpochLoss / len(train_data_loader)))
            invTrainHistory.append(invLossFn(fwdModel(invModel(y_all_train)), y_all_train).item())
            invTestHistory.append(invLossFn(fwdModel(invModel(y_all_test)), y_all_test).item())
        print('-------------------------------------')
        # save model
        torch.save(invModel, "models/invModel.pt")
        # export loss history
        exportList('loss-history/invTrainHistory', invTrainHistory)
        exportList('loss-history/invTestHistory', invTestHistory)
    else:
        invModel = torch.load("models/invModel.pt")
        invModel.eval()





    #############---TESTING---##############
    x_test, y_test = next(iter(test_data_loader))

    with torch.no_grad():
        y_test_pred = fwdModel(x_test)
        x_test_pred = invModel(y_test)
        x_test_pred_uncorrected = x_test_pred.detach().clone()




        y_test = pd.DataFrame(y_test.numpy())
        x_test = pd.DataFrame(x_test.numpy())

        y_test_pred  = pd.DataFrame(y_test_pred.numpy())
        x_test_pred  = pd.DataFrame(x_test_pred.numpy())

        ind = 50
        x = range(0,1001)
        plt.figure()
        plt.plot(x, y_test.iloc[ind, :], x, y_test_pred.iloc[ind, :] )
        plt.title(fwdEpochs)

        '''
        ind = 100
        x = range(0,16)
        plt.figure()
        plt.plot(x, x_test.iloc[ind, :], x, x_test_pred.iloc[ind, :] )
        plt.title(fwdEpochs)
        plt.show()
        '''


        A = 1



        # fix values so that theta is not 0 or below thetaMin
        x_test_pred = correctionDirect(x_test_pred)
        y_test_pred_pred = fwdModel(x_test_pred)

        #############---POST PROC---##############
        print('\nR2 values:\n--------------------------------------------')
        print('Fwd test Y R2:', computeR2(y_test_pred, y_test), '\n')

        print('Inv test reconstruction Y R2:', computeR2(y_test_pred_pred, y_test), '\n')

        print('Inv test prediction X R2:', computeR2(x_test_pred, x_test))
        print('^^ Dont freak out; this is expected to be (very) low')
        print('--------------------------------------------\n')