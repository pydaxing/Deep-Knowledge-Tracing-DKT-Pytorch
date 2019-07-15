import numpy as np
import itertools

class DataReader():
    def __init__(self, train_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques

    def getTrainData(self):
        print('loading train data...')
        trainData = []
        zero = [0 for i in range(self.numofques * 2)]
        with open(self.train_path, 'r') as train:
            for len, ques, ans in itertools.zip_longest(*[train] * 3):
                len = int(len.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                slices = len//self.maxstep + (1 if len%self.maxstep > 0 else 0)
                for i in range(slices):
                    temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                    if len > 0:
                        if len >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = len
                        for j in range(l):
                            if ans[i*self.maxstep + j] == 1:
                                temp[j][ques[i*self.maxstep + j]] = 1
                            else:
                                temp[j][ques[i*self.maxstep + j] + self.numofques] = 1
                        len = len - self.maxstep
                    trainData.append(temp.tolist())
            print('done: ' + str(np.array(trainData).shape))
            return np.array(trainData)

    def getTestData(self):
        print('loading test data...')
        testData = []
        zero = [0 for i in range(self.numofques * 2)]
        with open(self.test_path, 'r') as test:
            for len, ques, ans in itertools.zip_longest(*[test] * 3):
                len = int(len.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                slices = len // self.maxstep + (1 if len % self.maxstep > 0 else 0)
                for i in range(slices):
                    temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                    if len > 0:
                        if len >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = len
                        for j in range(l):
                            if ans[i * self.maxstep + j] == 1:
                                temp[j][ques[i * self.maxstep + j]] = 1
                            else:
                                temp[j][ques[i * self.maxstep + j] + self.numofques] = 1
                        len = len - self.maxstep
                    testData.append(temp.tolist())
            print('done: ' + str(np.array(testData).shape))
            return np.array(testData)

