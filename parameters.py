import torch

#s_par_types = ['SZmax_Zmin', 'SZmin_Zmin']





featureDim = 16
featureName = list(range(0,featureDim))

append_mask = 'mask_'
featureName = [append_mask + str(sub) for sub in featureName]


labelDim = 1001
labelNames = list(range(0,labelDim))
append_s11 = 's11_'
labelNames = [append_s11 + str(sub) for sub in labelNames]
#labelNames = [ s_par_type +'_'+ x for x in labelNames]


trainSplit = 0.9
testSplit = 0.1


batchSize = 64
batchShuffle = True
numWorkers = 0

randomWeights = False

fwdTrain = True
fwdEpochs =  1000
fwdHiddenDim = [32, 32, 64, 64, 128, 128, 512]
fwdHiddenLayers = len(fwdHiddenDim)-1
fwdLearningRate = 1e-4
fwdLossFn = torch.nn.MSELoss()

invTrain = True
invEpochs =  1000
invHiddenDim = [500, 100, 100, 64, 64,32,100]
invHiddenLayers = len(invHiddenDim)-1
invLearningRate = 1e-4
invLossFn = torch.nn.MSELoss()
betaX = 0.5
betaXEpochSchedule = 40
thetaMin = 0.1667 #normalized equivalent of 15 degrees