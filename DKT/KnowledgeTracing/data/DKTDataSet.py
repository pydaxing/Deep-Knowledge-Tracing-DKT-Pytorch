import numpy as np
from torch.utils.data.dataset import Dataset
from Constant import Constants as C
import torch

class DKTDataSet(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        onehot = self.onehot(questions, answers)
        return torch.FloatTensor(onehot.tolist())

    def onehot(self, questions, answers):
        result = np.zeros(shape=[C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS])
        for i in range(C.MAX_STEP):
            if answers[i] > 0:
                result[i][questions[i]] = 1
            elif answers[i] == 0:
                result[i][questions[i] + C.NUM_OF_QUESTIONS] = 1
        return result