import torch
from PIL import Image
import numpy as np

from cnsproject.plotting.plotting import plotting
from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.encoding.encoders import PoissonEncoder
from cnsproject.network.connections import Connection, fullyNormalConnect
from cnsproject.learning.learning_rules import STDP

time = 200
scale = 100
dt = 1
shape1 = (20,20)
shape2 = (2,)

im1 = Image.open("./example/images/img1.png").convert('L')
im2 = Image.open("./example/images/img3.jpg").convert('L')
im1 = im1.resize(shape1, Image.ANTIALIAS)
im2 = im2.resize(shape1, Image.ANTIALIAS)
s1 = torch.from_numpy(np.asarray(im1).copy())
s2 = torch.from_numpy(np.asarray(im2).copy())

encoder = PoissonEncoder(time = int(time/4), r=10)
I1 = encoder(s1)
I2 = encoder(s2)

I = torch.cat((I1,I2,I1,I2), 0)


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

con1 = Connection(
        pre = n1, post = n2, lr = None, weight_decay = 0.005,
        J = 2, tau_s = 10, trace_scale = 15., dt = dt, connectivity = fullyNormalConnect, wmean=5., wstd=.5,
        learning_rule=STDP, beta=1., gamma=.2
    )


monitor1 = Monitor(n1, state_variables=["s"])
monitor1.set_time_steps(time, dt)
monitor1.reset_state_variables()

monitor2 = Monitor(n2, state_variables=["s"])
monitor2.set_time_steps(time, dt)
monitor2.reset_state_variables()

monitor3 = Monitor(con1, state_variables=["w"])
monitor3.set_time_steps(time, dt)
monitor3.reset_state_variables()

I_ex = 0
for i in range(len(I)):
    n1.forward(I[i])
    n2.forward(I_ex)
    I_ex = con1.compute()
    con1.update(learning=True,)
    monitor1.record()
    monitor2.record()
    monitor3.record()


s1 = monitor1.get("s").flatten(start_dim=1)
s2 = monitor2.get("s").flatten(start_dim=1)
w = monitor3.get("w").flatten(start_dim=1)
I = I.flatten(start_dim=1)

plot = plotting()

plot.plot_learning_init(time/scale)
plot.plot_learning_update(s1.T, s2.T, w)
plot.show()

