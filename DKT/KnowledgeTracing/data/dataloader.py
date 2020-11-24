import sys
sys.path.append('../')
import torch
import torch.utils.data as Data
from Constant import Constants as C
from data.readdata import DataReader
from data.DKTDataSet import DKTDataSet

def getTrainLoader(train_data_path):
    handle = DataReader(train_data_path ,C.MAX_STEP, C.NUM_OF_QUESTIONS)
    trainques, trainans = handle.getTrainData()
    dtrain = DKTDataSet(trainques, trainans)
    trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True)
    return trainLoader

def getTestLoader(test_data_path):
    handle = DataReader(test_data_path, C.MAX_STEP, C.NUM_OF_QUESTIONS)
    testques, testans = handle.getTestData()
    dtest = DKTDataSet(testques, testans)
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False)
    return testLoader

def getLoader(dataset):
    trainLoaders = []
    testLoaders = []
    if dataset == 'assist2009':
        trainLoader = getTrainLoader(C.Dpath + '/assist2009/builder_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2009/builder_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'assist2015':
        trainLoader = getTrainLoader(C.Dpath + '/assist2015/assist2015_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2015/assist2015_test.txt')
        testLoaders.append(testLoader)
    elif dataset == 'static2011':
        trainLoader = getTrainLoader(C.Dpath + '/statics2011/static2011_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/statics2011/static2011_test.txt')
        testLoaders.append(testLoader)
    elif dataset == 'kddcup2010':
        trainLoader = getTrainLoader(C.Dpath + '/kddcup2010/kddcup2010_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/kddcup2010/kddcup2010_test.txt')
        testLoaders.append(testLoader)
    elif dataset == 'assist2017':
        trainLoader = getTrainLoader(C.Dpath + '/assist2017/assist2017_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2017/assist2017_test.txt')
        testLoaders.append(testLoader)
    elif dataset == 'synthetic':
        trainLoader = getTrainLoader(C.Dpath + '/synthetic/synthetic_train_v0.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/synthetic/synthetic_test_v0.txt')
        testLoaders.append(testLoader)
    return trainLoaders, testLoaders