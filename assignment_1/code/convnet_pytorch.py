"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()

    kernel_size = (3,3)
    padding = 1

    conv1 = nn.Conv2d(n_channels, 64, kernel_size, stride=1, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
    maxPool1 = nn.MaxPool2d(kernel_size, stride=2, padding=padding, dilation=1, return_indices=False, ceil_mode=False)
    batchNorm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    conv_layer1 = [conv1, maxPool1, batchNorm1, nn.ReLU()]

    conv2 = nn.Conv2d(64, 128, kernel_size, stride=1, padding=padding)
    maxPool2 = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
    batchNorm2 = nn.BatchNorm2d(128)
    conv_layer2 = [conv2, maxPool2, batchNorm2, nn.ReLU()]

    conv3_a = nn.Conv2d(128, 256, kernel_size, stride=1, padding=padding)
    conv3_b = nn.Conv2d(256, 256, kernel_size, stride=1, padding=padding)
    maxPool3 = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
    batchNorm3 = nn.BatchNorm2d(256)
    conv_layer3 = [conv3_a, conv3_b, maxPool3, batchNorm3, nn.ReLU()]

    conv4_a = nn.Conv2d(256, 512, kernel_size, stride=1, padding=padding)
    conv4_b = nn.Conv2d(512, 512, kernel_size, stride=1, padding=padding)
    maxPool4 = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
    batchNorm4 = nn.BatchNorm2d(512)
    conv_layer4 = [conv4_a, conv4_b, maxPool4, batchNorm4, nn.ReLU()]

    conv5_a = nn.Conv2d(512, 512, kernel_size, stride=1, padding=padding)
    conv5_b = nn.Conv2d(512, 512, kernel_size, stride=1, padding=padding)
    maxPool5 = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
    batchNorm5 = nn.BatchNorm2d(512)
    conv_layer5 = [conv5_a, conv5_b, maxPool5, batchNorm5, nn.ReLU()]

    flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)
    output_layer = nn.Linear(512, n_classes)

    self.layers = [*conv_layer1, *conv_layer2, *conv_layer3, *conv_layer4, *conv_layer5, flatten_layer, output_layer]
    self.model = nn.Sequential(*self.layers)
    print(self.model)
    #######################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.model(x)
    #######################
    # END OF YOUR CODE    #
    #######################

    return out
