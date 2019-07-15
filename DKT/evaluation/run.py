import sys
sys.path.append('../')
from model.RNNModel import RNNModel
from data.dataloader import getDataLoader
from Constant import Constants as C
import torch.optim as optim
from evaluation import eval

trainLoader, testLoade = getDataLoader()

rnn = RNNModel(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT)
optimizer = optim.Adam(rnn.parameters(), lr=C.LR)
loss_func = eval.lossFunc()

for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    rnn, optimizer = eval.train_epoch(rnn, trainLoader, optimizer, loss_func)
    eval.test_epoch(rnn, testLoade, loss_func)
