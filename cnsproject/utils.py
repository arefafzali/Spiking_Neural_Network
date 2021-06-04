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
        I[2*scale:, i] = I_value * (i+1)
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


def noise_function(time: int, neuron_size: int = 1, start: float = 200):
    I = np.zeros([time, neuron_size])+start
    for i in range(time-1):
        I[i, :] = abs(I[i-1, :]+np.random.normal(0, 1, size=(neuron_size))+np.random.normal(0, 2))
    I = torch.from_numpy(I)
    return I


def get_gaussian_kernel(k=3, sigma=1):
    steps = np.linspace(-1, 1, k)
    x, y = np.meshgrid(steps, steps)
    gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / (2 * np.pi *sigma **2) ** 0.5
    # gaussian = gaussian / np.sum(gaussian)
    return gaussian


def DoG_kernel(k=3, sigma1=0.3, sigma2=1):
    g1 = get_gaussian_kernel(k, sigma1)
    g2 = get_gaussian_kernel(k, sigma2)
    DoG = g1 - g2
    DoG -= DoG.mean()
    return DoG


def gabor_kernel(k=5, sigma=.4, teta=0, gamma=0.3, lambdal=0.8):
    step = np.linspace(-1, 1, k)
    x, y = np.meshgrid(step, step)
    X = x*np.cos(teta) + y*np.sin(teta)
    Y = -x*np.sin(teta) + y*np.cos(teta)
    gabor = np.exp(-(X**2 + (gamma**2) * (Y**2)) / (2 * sigma ** 2))
    gabor = gabor * np.cos(2*np.pi*X/lambdal)
    gabor -= gabor.mean()
    return gabor

def convolution(image, kernel, bias):
    m, n = kernel.shape
    y, x = image.shape
    y -= m + 1
    x -= n + 1
    new_image = torch.zeros((y,x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = torch.sum(image[i:i+m, j:j+n]*kernel) + bias
    return new_image