import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegularized(nn.Module):
    """
    Combined Layer of Batch Norm and Dropout and Fully Connected
    """

    def __init__(self, ins, outs, dropout_rate=0.3):
        super(LinearRegularized, self).__init__()
        self.fc = nn.Linear(ins, outs)
        self.bn = nn.BatchNorm1d(outs)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop(x)
        return x


class DropNN(nn.Module):
    """
    Small Network with Dropout regularization
    """

    def __init__(self):
        super(DropNN, self).__init__()
        self.fc1 = LinearRegularized(320, 100)
        self.fc2 = LinearRegularized(100, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x, contexts):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()


class ConditionLayer(nn.Module):
    def __init__(self, ins, outs, context, dropout_rate=0.3, non_linear=True):
        super(ConditionLayer, self).__init__()
        self.ins, self.outs, self.contexts = ins, outs, context
        self.non_linear = non_linear

        self.fc = nn.Linear(ins, outs)
        self.affine_W = nn.Embedding(context, outs ** 2)
        self.affine_b = nn.Embedding(context, outs)
        self.batch_norm = nn.BatchNorm1d(outs)
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, sample, context):
        h = self.fc(sample).unsqueeze(-1)
        W = self.affine_W(context).reshape(-1, self.outs, self.outs)
        b = self.affine_b(context).unsqueeze(-1)
        z = self.batch_norm(W @ h + b)
        a = F.relu(z) if self.non_linear else z
        return self.drop_out(a.squeeze())


class ConditionNet(nn.Module):
    def __init__(self):
        super(ConditionNet, self).__init__()
        self.cl1 = ConditionLayer(ins=320, outs=100, context=3)
        self.cl2 = ConditionLayer(ins=100, outs=40, context=3)
        self.cl3 = ConditionLayer(ins=40, outs=1, context=3, dropout_rate=0, non_linear=False)

    def forward(self, sample, context):
        h = self.cl1(sample, context)
        h = self.cl2(h, context)
        h = torch.sigmoid(self.cl3(h, context))
        return h.squeeze()
