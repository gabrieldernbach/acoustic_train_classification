import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Flatten(nn.Module):
    def forward(self, input):
        # return input.squeeze()
        return input.view(input.size(0), -1)


"""
Modified Efficient Net (Key Feature : Scalability)
The network was originally specified for imagenet which
takes 3 x 224 x 224 images as inputs and predicts 1000 classes.

The input resolution of the spectrograms is adapted by resizing.
Input channels get expanded by prepending an additional convolutional layer that project to 3 channels
For the outputs we prepend one fully connected layer mapping to 1000 nodes to 1.
"""
eff_net = EfficientNet.from_name('efficientnet-b0')
first_conv_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
eff_net = nn.Sequential(first_conv_layer, eff_net, nn.Linear(1000, 1), Flatten())


# squeezenet = torch.hub.load('pytorch/vision:v0.4.2', 'squeezenet1_0', pretrained=True)
# first_conv_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
# squeezenet = nn.Sequential(first_conv_layer, squeezenet, nn.Linear(1000, 1, Flatten()))

