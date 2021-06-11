import torch
from PIL import Image
import numpy as np

from cnsproject.plotting.plotting import plotting
from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.encoding.encoders import PoissonEncoder, Time2FirstSpikeEncoder
from cnsproject.network.connections import ConvolutionalConnection, PoolingConnection
from cnsproject.utils import gabor_kernel

time = 200
scale = 100
dt = 1
shape1 = (30,30)
shape2 = (22,22)
shape3 = (21,21)


im = Image.open("./example/images/img1.png").convert('L')
im = im.resize(shape1, Image.ANTIALIAS)
s = torch.from_numpy(np.asarray(im).copy())


# encoder = Time2FirstSpikeEncoder(time = time)
encoder = PoissonEncoder(time = time, r=10)
I = encoder(s)


kernel = gabor_kernel(k=7, sigma=.4, teta=0, gamma=.7, lambdal=.8)


n1 = InputPopulation(
        shape = shape1, spike_trace = True, additive_spike_trace = True, tau_s = 4, trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 1, C = 20, threshold = -40
    )
n1.dt = dt

n2 = LIFPopulation(
        shape = shape2, spike_trace = True, additive_spike_trace = True, tau_s = 4, trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 1, C = 20, threshold = -40
    )
n2.dt = dt

n3 = LIFPopulation(
        shape = shape3, spike_trace = True, additive_spike_trace = True, tau_s = 4, trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 1, C = 20, threshold = -40
    )
n3.dt = dt

con1 = ConvolutionalConnection(
        pre = n1, post = n2, kernel = kernel, J = 1000
    )

con2 = PoolingConnection(
        pre = n2, post = n3, k = 2, J = 1000
    )


monitor1 = Monitor(n1, state_variables=["s"])
monitor1.set_time_steps(time, dt)
monitor1.reset_state_variables()

monitor2 = Monitor(n2, state_variables=["s"])
monitor2.set_time_steps(time, dt)
monitor2.reset_state_variables()


I1 = 0
I2 = 0
for i in range(len(I)):
    n1.forward(I[i])
    n2.forward(I1)
    n3.forward(I2)
    I1 = con1.compute()
    I2 = con2.compute()
    monitor1.record()
    monitor2.record()

s1 = monitor1.get("s").flatten(start_dim=1)
s2 = monitor2.get("s").flatten(start_dim=1)
I = I.flatten(start_dim=1)

d1 = encoder.decode(s1.numpy(), shape1)
d2 = encoder.decode(s2.numpy(), shape2)


plot = plotting()

plot.plot_visual_activity(s2.T, d1, d2)
plot.show()

