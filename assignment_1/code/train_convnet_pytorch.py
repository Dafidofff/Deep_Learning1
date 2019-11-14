"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None
dtype = torch.FloatTensor

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).type(dtype).mean().item()
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def create_new_text(name, train_list, test_list, list_params, loss_list):
  tfile = open(f'../{name}.txt', 'w+')
  tfile.write(f'{train_list}')
  tfile.write(f'{test_list}')
  tfile.write(f'{loss_list}')
  tfile.write(f'{list_params}')
  tfile.close()

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  train_accs, train_losses, test_accs, test_losses = [], [], [], []
  cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

  train_x = cifar10['train'].images
  train_x = torch.from_numpy(train_x).type(dtype)
  train_y = cifar10['test'].labels
  train_y = torch.from_numpy(train_y).type(dtype)

  test_x = cifar10['test'].images
  test_x = torch.from_numpy(test_x).type(dtype)
  test_y = cifar10['test'].labels
  test_y = torch.from_numpy(test_y).type(dtype)

  convNet = ConvNet(3,10)
  optimizer = optim.Adam(convNet.model.parameters(), lr = FLAGS.learning_rate, weight_decay=1e-6)
  loss_function = nn.CrossEntropyLoss()

  for epoch in range(FLAGS.max_steps):
      convNet.train()

      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      x = torch.from_numpy(x).type(dtype)
      y = torch.from_numpy(y).type(dtype)

      optimizer.zero_grad()
      out = convNet.forward(x)

      loss = loss_function(out, y.argmax(1))
      loss.backward()
      optimizer.step()
      loss.retain_grad()

      convNet.eval()
      # Save losses for every 100th epoch
      if epoch % FLAGS.eval_freq == 0 :
          train_out = convNet.forward(x)
          test_out = convNet.forward(test_x)

          train_acc = accuracy(train_out, y)
          test_acc = accuracy(test_out, test_y)
          train_accs.append(train_acc)
          test_accs.append(test_acc)

          train_loss = loss_function.forward(train_out, y.argmax(1)).item()
          test_loss = loss_function.forward(test_out, test_y.argmax(1)).item()
          train_losses.append(train_loss)
          test_losses.append(test_loss)
          print('train error: ', train_loss, ' validation error: ', test_loss, ' validation accuracy: ', test_acc)
  
  # Save lists
  create_new_text("pytorch_1", train_accs, test_accs, train_losses, test_losses)

  # Plot everything
  fig = plt.figure()
  ax1 = fig.add_subplot(121)
  ax1.plot(train_accs, label='train accuracies' )
  ax1.plot(test_accs, label='test accuracies')
  ax1.legend()

  ax2 = fig.add_subplot(122)
  ax2.plot(train_losses, label='train losses' )
  ax2.plot(test_losses, label='test losses' )
  ax2.legend()
  # plt.show()
  plt.savefig("convnet_results")

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()