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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from part2.dataset import TextDataset
# from part2.model import TextGenerationModel
from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def calc_accuracy(predictions, targets):
  return (predictions.max(axis=1)[1].cpu().numpy() == targets.cpu().numpy()).sum()/predictions.shape[0]

def train(config):

    # Initialize the device which to run the model on
    # device = torch.device(config.device)
    config.device = "gpu"
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset("grims_fairy_tales.txt", config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers, config.device).to(config.device)  

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(config.device)
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=0.5)

    steps = 0
    while steps <= config.train_steps:
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()
            one_hot_batch = torch.nn.functional.one_hot(batch_inputs.to(torch.int64), dataset.vocab_size).to(config.device)
            batch_targets.to(config.device)

            optimizer.zero_grad()
            out = model.forward(one_hot_batch).to(config.device)
            
            loss = criterion(out, batch_targets)
            loss.backward()
            optimizer.step()
            loss.retain_grad()

            accuracy = calc_accuracy(out, batch_targets)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)


            if step % config.print_every == 0:
                print(f"({datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}), Train Step: {steps}/{config.train_steps}, Batch_size: {config.batch_size}, E/sec: {int(examples_per_second)}, Acc: {accuracy}, Loss: {float(loss)}")
                # print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                #       "Accuracy = , Loss = {:.3f}".format(
                #         datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                #         config.train_steps, config.batch_size, examples_per_second,
                #         accuracy, loss
                # ))
                # first_letter = dataset.convert_to_string([batch_inputs[0,0].item()])
                # target_string = dataset.convert_to_string([item.item() for item in batch_targets[0,:]])
                # target_string = target_string.replace('\n', ' ')

                # print(target_string)
                # print(f"original string: \"{dataset.convert_to_string(first_letter + batch_targets[0,:].tolist())}\"".replace('\n',''))
                # print(f"predicted string:\"{dataset.convert_to_string(first_letter + torch.max(predictions_test,2)[1][0,:].tolist())}\"".replace('\n',''))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                pass

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
            steps += 1
    torch.save(model.state_dict(), './models/Lorde1_.p')
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
