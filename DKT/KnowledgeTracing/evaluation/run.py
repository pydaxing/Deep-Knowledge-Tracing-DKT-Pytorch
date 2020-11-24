import sys
sys.path.append('../')
from model.RNNModel import DKT
from data.dataloader import getTrainLoader, getTestLoader, getLoader
from Constant import Constants as C
import torch.optim as optim
from evaluation import eval

print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')

model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT)
optimizer_adam = optim.Adam(model.parameters(), lr=C.LR)
optimizer_adgd = optim.Adagrad(model.parameters(),lr=C.LR)

loss_func = eval.lossFunc()

trainLoaders, testLoaders = getLoader(C.DATASET)

for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = eval.train(trainLoaders, model, optimizer_adgd, loss_func)
    eval.test(testLoaders, model)
