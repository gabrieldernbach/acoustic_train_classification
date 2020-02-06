import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegularized(nn.Module):
    """
    Combined Layer of Batch Norm and Dropout and Fully Connected
    """

    def __init__(self, ins, outs, dropout_rate=0.5):
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
        return x


class DropNNBig(nn.Module):
    def __init__(self):
        super(DropNNBig, self).__init__()
        self.fc1 = LinearRegularized(640, 320, dropout_rate=0.3)
        self.fc2 = LinearRegularized(320, 40, dropout_rate=0.3)
        self.fc3 = LinearRegularized(40, 1, dropout_rate=0)

    def forward(self, x, context):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class ConditionLayer(nn.Module):
    """
    Linear Layer with subsequent context dependent Affine Transformation
    """

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
    """
    Fully Connected Network that allows for a categorical context as conditional
    """

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


class ElementConditionLayer(nn.Module):
    """
    A layer that incorporates a context dependent multiplication 'w' and offset 'b'
    """

    def __init__(self, ins, outs, context, dropout_rate=0.3, non_linear=True):
        super(ElementConditionLayer, self).__init__()
        self.ins, self.outs, self.contexts = ins, outs, context
        self.non_linear = non_linear

        self.fc = nn.Linear(ins, outs)
        self.affine_w = nn.Embedding(context, outs)
        self.affine_b = nn.Embedding(context, outs)
        self.batch_norm = nn.BatchNorm1d(outs)
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, sample, context):
        h = self.fc(sample)
        w = self.affine_w(context)
        b = self.affine_b(context)
        z = self.batch_norm(w * h + b)
        a = F.relu(z) if self.non_linear else z
        return self.drop_out(a.squeeze())


class ElementConditionNet(nn.Module):
    """
    Fully Connected Network that allows for a categorial context as conditional,
    condigioning is applied layer wise by per activation multiplication and offset
    """

    def __init__(self):
        super(ElementConditionNet, self).__init__()
        self.cl1 = ElementConditionLayer(ins=320, outs=320, context=3)
        self.cl2 = ElementConditionLayer(ins=320, outs=200, context=3)
        self.cl3 = ElementConditionLayer(ins=200, outs=100, context=3)
        self.cl4 = ElementConditionLayer(ins=100, outs=40, context=3)
        self.cl5 = ElementConditionLayer(ins=40, outs=1, context=3, dropout_rate=0, non_linear=False)

    def forward(self, sample, context):
        h = self.cl1(sample, context)
        h = self.cl2(h, context)
        h = self.cl3(h, context)
        h = self.cl4(h, context)
        h = torch.sigmoid(self.cl5(h, context))
        return h.squeeze()
