from cnsproject.utils import DoG_kernel, convolution, gabor_kernel
from cnsproject.plotting.plotting import plotting
from cnsproject.encoding.encoders import PoissonEncoder, Time2FirstSpikeEncoder

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# kernel = DoG_kernel(k=5, sigma1=0.3, sigma2=1)
kernel = gabor_kernel(k=5, sigma=.4, teta=0, gamma=.3, lambdal=.8)

im = Image.open("./example/images/img.jpg").convert('L')
im = im.resize((200,200),Image.ANTIALIAS)
data = torch.from_numpy(np.asarray(im).copy())

result_image = convolution(data, kernel, 0)

time = 20
r = 10

encoder = PoissonEncoder(time = time, r=r)
# encoder = Time2FirstSpikeEncoder(time = time)
encoded_image = encoder(result_image)

fi = torch.flatten(encoded_image, start_dim=1)
decoded_image = encoder.decode(fi.numpy(), result_image.shape)

plot = plotting()
plot.plot_kernel_surface(kernel)
plot.plot_v1(im, kernel, result_image)
plot.show()
