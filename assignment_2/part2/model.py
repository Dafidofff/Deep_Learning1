# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

	def __init__(self, batch_size, seq_length, vocabulary_size,
				 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

		super(TextGenerationModel, self).__init__()

		# self.network = []
		# self.network.append(nn.LSTM(input_size = vocabulary_size, hidden_size = lstm_num_hidden, num_layers = lstm_num_layers).double())
		
		# input_size = vocabulary_size
		# output_size = lstm_num_hidden
		# for i in range(lstm_num_layers):
		# 	self.network.append(nn.LSTM(input_size, output_size).double())
		# 	input_size = lstm_num_hidden

		# self.network.append(nn.Linear(lstm_num_hidden, vocabulary_size).double())
		# self.model = nn.Sequential(*self.network)
		# print(self.model)

		self.LSTM = nn.LSTM(input_size = vocabulary_size, hidden_size = lstm_num_hidden, num_layers = lstm_num_layers, batch_first = True).double()
		self.linear = nn.Linear(lstm_num_hidden, vocabulary_size).double()

	def forward(self, x):
		x = x.to(torch.double)
		out_temp, states = self.LSTM(x)
		out = self.linear(out_temp)

		return out
