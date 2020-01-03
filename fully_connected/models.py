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
        self.fc3 = nn.Linear(40, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class OldConditionNet(nn.Module):
    def __init__(self, layer_shapes=None, n_context=3):
        super(OldConditionNet, self).__init__()
        if layer_shapes is None:
            layer_shapes = [2, 80, 2]

        self.layer_shapes = layer_shapes

        self.fc1 = nn.Linear(layer_shapes[0], layer_shapes[1])
        self.fc1_affineW = nn.Linear(n_context, layer_shapes[1] ** 2)
        self.fc1_affineB = nn.Linear(n_context, layer_shapes[1])

        self.fc2 = nn.Linear(layer_shapes[1], layer_shapes[2])
        self.fc2_affineW = nn.Linear(n_context, layer_shapes[2] ** 2)
        self.fc2_affineB = nn.Linear(n_context, layer_shapes[2])

    def forward(self, sample, context):
        h1 = self.fc1(sample).unsqueeze(-1)
        W1 = self.fc1_affineW(context).reshape(-1, self.layer_shapes[1], self.layer_shapes[1])
        b1 = self.fc1_affineB(context).unsqueeze(-1)
        a1 = F.relu(W1 @ h1 + b1).squeeze()

        h2 = self.fc2(a1).unsqueeze(-1)
        W2 = self.fc2_affineW(context).reshape(-1, 2, 2)
        b2 = self.fc2_affineB(context).unsqueeze(-1)
        a2 = (W2 @ h2 + b2).squeeze()

        return a2


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

    def forward(self, inputs):
        sample, context = inputs
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
        self.cl1 = ConditionLayer(ins=40, outs=2, context=3)
