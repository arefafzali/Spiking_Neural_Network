from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.plotting.plotting import plotting
from cnsproject.utils import step_function, noise_function, step_noise_function
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import Connection, randomNormalConnect

import torch
import numpy as np


time = 500
scale = 100
dt = 1
neuron_size = 100
shape1 = (int(neuron_size*0.8),)
shape2 = (int(neuron_size*0.2),)

# I1 = noise_function(time = time)
# I2 = I1

I1 = noise_function(time, shape1[0])
I2 = noise_function(time, shape2[0])


pn1 = LIFPopulation(
        shape = shape1, spike_trace = True, additive_spike_trace = True, tau_s = 10, trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 1, C = 20, threshold = -40, dt = dt
    )

pn2 = LIFPopulation(
        shape = shape2, spike_trace = True, additive_spike_trace = True, tau_s = 10, trace_scale = 1.,
        is_inhibitory = True, learning = False, R = 1, C = 20, threshold = -40, dt = dt
    )

con1 = Connection(
        pre = pn1, post = pn1, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=20., wstd=5.
    )

con2 = Connection(
        pre = pn2, post = pn1, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=21., wstd=5.
    )

con3 = Connection(
        pre = pn1, post = pn2, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=20., wstd=5.
    )

monitor1 = Monitor(pn1, state_variables=["s", "u"])
monitor1.set_time_steps(time, dt)
monitor1.reset_state_variables()

monitor2 = Monitor(pn2, state_variables=["s", "u"])
monitor2.set_time_steps(time, dt)
monitor2.reset_state_variables()

I_in=0
I_self=0
I_ex=0
for i in range(len(I1)):
    pn1.forward(I1[i] + I_self - I_in)
    pn2.forward(I2[i] + I_ex)
    I_self = con1.compute()
    I_in = con2.compute()
    I_ex = con3.compute()
    monitor1.record()
    monitor2.record()

s1 = torch.transpose(monitor1.get("s")*1, 0, 1)
s2 = torch.transpose(monitor2.get("s")*1, 0, 1)

plot = plotting()

plot.plot_population_activity_init(time/scale)
plot.plot_population_activity_update(s1, I1, mode="p1")
plot.plot_population_activity_update(s2, I2, start_idx=int(s1.numel()/s1.shape[-1]), mode="p2")
plot.show()


