from PIL import Image
import numpy as np
import torch

from cnsproject.encoding.encoders import PoissonEncoder
from cnsproject.plotting.plotting import plotting


im = Image.open("./example/images/img.jpg").convert('L')
im = im.resize((200,200),Image.ANTIALIAS)
data = torch.from_numpy(np.asarray(im))

time = 20
r = 10

encoder = PoissonEncoder(time = time, r=r)

I = encoder(data)

fi = torch.flatten(I, start_dim=1)

x = encoder.decode(fi.numpy(), (200,200))

plot = plotting()
plot.plot_encoding_decoding(data,fi,x)
plot.show()

