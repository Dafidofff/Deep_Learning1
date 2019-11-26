################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import argparse
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from part1.dataset import PalindromeDataset
# from part1.vanilla_rnn import VanillaRNN
# from part1.lstm import LSTM

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

import sys


# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def evaluate_model(config):
	seeds = [13,69,420]
	palindrome_lengths = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
	accuracies = []
	for i, seed in enumerate(seeds):
		random.seed(seed)
		accuracies.append([])
		for p_len in palindrome_lengths:
			config.seq_length = p_len

			# Train the model
			accuracies[i].append(train(config, print_eval=False))
	print(accuracies)
	return accuracies


def calc_accuracy(predictions, targets):
  return (predictions.max(axis=1)[1].cpu().numpy() == targets.cpu().numpy()).sum()/predictions.shape[0]
  

def train(config, print_eval = True):
	assert config.model_type in ('RNN', 'LSTM')

	# Initialize the device which to run the model on
	device = torch.device(config.device)

	# Initialize the model that we are going to use
	if config.model_type == "RNN":
		model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes)
	else:
		model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes)

	# Initialize the dataset and data loader (note the +1)
	dataset = PalindromeDataset(config.input_length+1)
	data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

	# Setup the loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.RMSprop(model.parameters(), alpha = 0.99, eps = 1e-6, lr=config.learning_rate, weight_decay = 0.1, momentum = 0.8)

	for step, (batch_inputs, batch_targets) in enumerate(data_loader):

		# Only for time measurement of step through network
		t1 = time.time()
		batch_inputs = torch.nn.functional.one_hot(batch_inputs.to(torch.int64),10)
		optimizer.zero_grad()
		out = model.forward(batch_inputs)

		############################################################################
		# QUESTION: what happens here and why?
		############################################################################
		torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
		############################################################################

		loss = criterion(out, batch_targets)
		loss.backward()
		optimizer.step()
		loss.retain_grad()

		accuracy = calc_accuracy(out, batch_targets)

		# Just for time measurement
		t2 = time.time()
		examples_per_second = config.batch_size/float(t2-t1)

		if step % 10 == 0 and print_eval:

			print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
				  "Accuracy = {:.2f}, Loss = {:.3f}".format(
					datetime.now().strftime("%Y-%m-%d %H:%M"), step,
					config.train_steps, config.batch_size, examples_per_second,
					accuracy, loss
			))

		# Check if model is converged, based on 5 batches.
		if step % 100 == 0 and step > config.input_length*70:
			test_accuracies = []
			for counter, (batch_inputs, batch_targets) in enumerate(data_loader):
				batch_inputs = torch.nn.functional.one_hot(batch_inputs.to(torch.int64),10)
				out = model.forward(batch_inputs)
				test_accuracies.append(calc_accuracy(out, batch_targets))
				if counter > 5: 
					break

			if np.std(test_accuracies) < 0.01:
				mean_acc = np.mean(test_accuracies)

				print('Training converged with accuracy:', mean_acc)
				return mean_acc




		if step == config.train_steps:
			# If you receive a PyTorch data-loader error, check this bug report:
			# https://github.com/pytorch/pytorch/pull/9655
			break

	print('Done training.')
	return accuracy

 ################################################################################
 ################################################################################

if __name__ == "__main__":
	# Parse training configuration
	parser = argparse.ArgumentParser()

	# Model params
	parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
	parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
	parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')
	parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
	parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
	parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
	parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
	parser.add_argument('--max_norm', type=float, default=10.0)
	parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

	parser.add_argument('--eval', type=bool, default=False, help="Checks if evaluation is needed")
	config = parser.parse_args()

	# Train the model
	if config.eval:
		evaluate_model(config)
	else:	
		train(config)
