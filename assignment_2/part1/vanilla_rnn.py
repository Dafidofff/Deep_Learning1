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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

	def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
		super(VanillaRNN, self).__init__()

		self.seq_length = seq_length
		self.device = device

		self.W_hx = nn.Parameter(torch.randn(input_dim, num_hidden).to(torch.float64))
		self.W_hh = nn.Parameter(torch.randn(num_hidden, num_hidden).to(torch.float64))
		self.W_ph = nn.Parameter(torch.randn(num_hidden, num_classes).to(torch.float64))
		self.B_h = nn.Parameter(torch.zeros(num_hidden).to(torch.float64))
		self.B_p = nn.Parameter(torch.zeros(num_classes).to(torch.float64))

		torch.nn.init.xavier_uniform_(self.W_hx)
		torch.nn.init.xavier_uniform_(self.W_hh)
		torch.nn.init.xavier_uniform_(self.W_ph)

		self.h_0 = torch.zeros(num_hidden).to(torch.float64)

	def forward(self, x):
		h_t = self.h_0
		x = x.to(torch.float64)
		self.all_gradients = []

		for i,step in enumerate(range(self.seq_length)):
			h_t = torch.tanh(x[:,step,:] @ self.W_hx + h_t @ self.W_hh + self.B_h)
			self.all_gradients.append(h_t.requires_grad_(True))
		return (h_t @ self.W_ph).add(self.B_p)

