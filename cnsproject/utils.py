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
    I[5*scale:10*scale, :] = I_value
    return I


def random_step_function(time: int, I_value: int, scale: int, neuron_size: int = 1):
    I = np.zeros([time, neuron_size])
    n = np.random.randint(time-20, size=10)
    n = np.sort(np.unique(n))
    for i in n:
        I[i:i+20, :] = np.random.randint(I_value)
    I = torch.from_numpy(I)
    return I

def incremental_step_noise_function(time: int, I_value: int, scale: int, neuron_size: int = 1, gap = 5):
    I = np.zeros([time, neuron_size])+I_value
    for i in range(int((time/scale)/gap)):
        x = I_value / (gap*(i+1)*scale - (gap*(i+1)-1)*scale)
        for j in range((gap*(i+1)-1)*scale, gap*(i+1)*scale):
            I[j, :] = x + I[j-1,:]
        I[gap*(i+1)*scale:, :] += I_value
    for i in range(time-1):
        I[i, :] += np.random.normal(0, 1, size=(neuron_size))+np.random.normal(0, 1)
    I = torch.from_numpy(I)
    return I

def step_noise_function(time: int, I_value: int, scale: int, neuron_size: int = 1):
    I = torch.normal(I_value, 1, size=(time, neuron_size))
    return I


def noise_function(time: int, neuron_size: int = 1):
    I = np.zeros([time, neuron_size])+200
    for i in range(time-1):
        I[i, :] = abs(I[i-1, :]+np.random.normal(0, 1, size=(neuron_size))+np.random.normal(0, 2))
    I = torch.from_numpy(I)
    return I

