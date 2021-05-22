from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from cnsproject.encoding.encoders import PositionEncoder
from cnsproject.plotting.plotting import plotting

im = Image.open("./example/encoding/img.jpg").convert('L')
im = im.resize((200,200),Image.ANTIALIAS)
data = torch.from_numpy(np.asarray(im))

time = 20
neuron_numbers = 8
neuron_range_mean = torch.from_numpy(np.arange(1,neuron_numbers+1))
neuron_range_std = torch.tensor([1]*neuron_numbers)


encoder = PositionEncoder(time = time, neuron_numbers=neuron_numbers,
    neuron_range_mean=neuron_range_mean, neuron_range_std=neuron_range_std)

I = encoder(data)
x=encoder.decode(I)

fi = torch.flatten(I, start_dim=1)


plot = plotting()
plot.plot_encoding_decoding(data,fi,x)
plot.show()
