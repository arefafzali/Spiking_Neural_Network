from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from cnsproject.encoding.encoders import PositionEncoder

im = Image.open("./example/encoding/img.jpg").convert('L')
im = im.resize((200,200),Image.ANTIALIAS)
data = torch.from_numpy(np.asarray(im))

time = 20
neuron_numbers = 8
neuron_range_mean = np.arange(1,neuron_numbers+1)
neuron_range_std = [1]*neuron_numbers

encoder = PositionEncoder(time = time, neuron_numbers=neuron_numbers,
    neuron_range_mean=neuron_range_mean, neuron_range_std=neuron_range_std)

I = encoder(data)

fi = torch.flatten(I, start_dim=1)

sf = np.flipud(fi.T)
args = np.argwhere(sf)
plt.scatter(args.T[1,:], args.T[0,:], s=0.1)
plt.show()

