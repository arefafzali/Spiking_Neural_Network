from PIL import Image
import numpy as np
import torch

from cnsproject.encoding.encoders import Time2FirstSpikeEncoder
from cnsproject.plotting.plotting import plotting


im = Image.open("./example/images/img.jpg").convert('L')
im = im.resize((200,200),Image.ANTIALIAS)
s = torch.from_numpy(np.asarray(im))

time = 20
encoder = Time2FirstSpikeEncoder(time = time)

I = encoder(s)

fi = torch.flatten(I, start_dim=1)

x = encoder.decode(fi.numpy(), (200,200))

plot = plotting()
plot.plot_encoding_decoding(s,fi,x)
plot.show()
