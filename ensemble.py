import torch.nn as nn
import torch

class Ensemble(nn.Module):

    def __init__(self, modelA, modelB, modelC):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = out1 + out2 + out3
        return out / 3
