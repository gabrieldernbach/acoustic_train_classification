import torch
import torch.nn as nn


class GaussianLayer(nn.Module):
    def __init__(self, ins, outs):
        super(GaussianLayer, self).__init__()
        self.fc_mu = nn.Linear(ins, outs)
        self.fc_logvar = nn.Linear(ins, outs)

    def forward(self, inputs):
        mu = self.fc_mu(inputs)

        logvar = self.fc_lgovar(inputs)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(),
            GaussianLayer(400, 20)
        )

        self.decode = nn.Sequential(
            nn.Linear(20, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid()
        )

    def forward(self, inputs):
        z, mu, logvar = self.encode(inputs)
        out = self.decode(z)
        return out, mu, logvar


class CClassifier(nn.Module):
    def __init__(self):
        super(CClassifier, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(),
            GaussianLayer(400, 200)
        )

        self.classifier = nn.Sequential(
            nn.Linear(200, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.Sigmoid()
        )

    def forward(self, inputs):
        z, mu, logvar = self.encode(inputs)
        out = self.decode(mu)
        return out, mu, logvar

# class ContextVAE(nn.Module):
#     def __init__(self, n_contexts):
#         super(ContextVAE, self).__init__()
#
#         self.encode = nn.Sequential(
#             nn.Linear(784, 400), nn.ReLU(),
#             nn.Linear(400, 200), nn.ReLU()
#         )
#
#         self.parameterize_z = nn.Sequential(
#             nn.Linear(200, n_contexts),
#             nn.Softmax(n_contexts)
#         )
#
#         self.parameterize_h = nn.Sequential(
#             nn.GaussianLayer
#         )
