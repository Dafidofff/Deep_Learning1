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

class LSTM(nn.Module):

	def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
		super(LSTM, self).__init__()
		
		self.W_gx = nn.Parameter(torch.randn(input_dim, num_hidden).to(torch.float64))
		self.W_ix = nn.Parameter(torch.randn(input_dim, num_hidden).to(torch.float64))
		self.W_fx = nn.Parameter(torch.randn(input_dim, num_hidden).to(torch.float64))
		self.W_ox = nn.Parameter(torch.randn(input_dim, num_hidden).to(torch.float64))

		self.W_gh = nn.Parameter(torch.randn(num_hidden, num_hidden).to(torch.float64))
		self.W_ih = nn.Parameter(torch.randn(num_hidden, num_hidden).to(torch.float64))
		self.W_fh = nn.Parameter(torch.randn(num_hidden, num_hidden).to(torch.float64))
		self.W_oh = nn.Parameter(torch.randn(num_hidden, num_hidden).to(torch.float64))

		self.B_g = nn.Parameter(torch.randn(num_hidden).to(torch.float64))
		self.B_i = nn.Parameter(torch.randn(num_hidden).to(torch.float64))
		self.B_f = nn.Parameter(torch.zeros(num_hidden).to(torch.float64))
		self.B_o = nn.Parameter(torch.randn(num_hidden).to(torch.float64))
		
		self.W_ph = nn.Parameter(torch.randn(num_hidden, num_classes).to(torch.float64))
		self.B_p = nn.Parameter(torch.randn(num_classes).to(torch.float64))

		self.h_0 = torch.zeros(num_hidden).to(torch.float64)
		self.c_0 = torch.zeros(num_hidden).to(torch.float64)
		self.seq_length = seq_length

		torch.nn.init.xavier_uniform_(self.W_gx)
		torch.nn.init.xavier_uniform_(self.W_ix)
		torch.nn.init.xavier_uniform_(self.W_fx)
		torch.nn.init.xavier_uniform_(self.W_ox)
		torch.nn.init.xavier_uniform_(self.W_gh)
		torch.nn.init.xavier_uniform_(self.W_ih)
		torch.nn.init.xavier_uniform_(self.W_fh)
		torch.nn.init.xavier_uniform_(self.W_oh)
		torch.nn.init.xavier_uniform_(self.W_ph)

	def forward(self, x):
		h_t = self.h_0
		c_t = self.c_0
		x = x.to(torch.float64)
		self.all_gradients = []

		for step in range(self.seq_length):
			g_t = torch.tanh(x[:,step,:] @ self.W_gx + h_t @ self.W_gh + self.B_g)
			i_t = torch.sigmoid(x[:,step,:] @ self.W_ix + h_t @ self.W_ih + self.B_i)
			f_t = torch.sigmoid(x[:,step,:] @ self.W_fx + h_t @ self.W_fh + self.B_f)
			o_t = torch.sigmoid(x[:,step,:] @ self.W_ox + h_t @ self.W_oh + self.B_o)

			c_t = g_t * i_t + c_t * f_t
			h_t = torch.tanh(c_t) * o_t
			self.all_gradients.append(h_t.requires_grad_(True))

		return (h_t @ self.W_ph).add(self.B_p)
