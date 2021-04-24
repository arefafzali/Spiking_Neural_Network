"""
Module for utility functions.

Note: You are going to need to implement DoG and Gabor filters. A possible opt
ion would be to write them in this file but it is not a must and you can define\
a separate module/package for them.
"""
import torch
import numpy as np


def step_function(time: int, I_value: int, scale: int, neuron_size: int = 1):
    I = torch.zeros(time, neuron_size)
    for i in range(neuron_size):
        I[5*scale:, i] = I_value * (i+1)
    return I


def two_way_step_function(time: int, I_value: int, scale: int, neuron_size: int = 1):
    I = torch.zeros(time, neuron_size)
    I[5*scale:10*scale, 0] = I_value
    return I


def random_step_function(time: int, I_value: int, scale: int, neuron_size: int = 1):
    I = np.zeros([time, neuron_size])
    n = np.random.randint(time-20, size=10)
    n = np.sort(np.unique(n))
    for i in n:
        I[i:i+20, 0] = np.random.randint(I_value)
    I = torch.from_numpy(I)
    return I

def random2_step_function(time: int, I_value: int, scale: int, neuron_size: int = 1):
    I = torch.rand(time, neuron_size)*I_value
    return I
