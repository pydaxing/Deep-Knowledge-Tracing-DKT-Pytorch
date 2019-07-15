import sys
sys.path.append('../')
import torch
import torch.utils.data as Data
from Constant import Constants as C
from data.readdata import DataReader

def getDataLoader():
    handle = DataReader('../dataset/assistments/builder_train.csv', '../dataset/assistments/builder_test.csv',C.MAX_STEP, C.NUM_OF_QUESTIONS)
    dtrain = torch.FloatTensor(handle.getTrainData().astype(float).tolist())
    dtest = torch.FloatTensor(handle.getTestData().astype(float).tolist())
    trainLoader = Data.DataLoader(dtrain, batch_size= C.BATCH_SIZE, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size= C.BATCH_SIZE, shuffle=False)
    return trainLoader, testLoader
