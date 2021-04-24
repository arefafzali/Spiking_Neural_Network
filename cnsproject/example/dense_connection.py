from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.plotting.plotting import plotting
from cnsproject.utils import step_function, two_way_step_function, random_step_function, random2_step_function
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection

import torch
import numpy as np


time = 1500
scale = 100
dt = 1
neuron_size = 10
shape1 = (int(neuron_size*0.8),)
shape2 = (int(neuron_size*0.2),)

I1 = step_function(time = time, I_value = 3, scale = scale)
# I2 = random_step_function(time = time, I_value = 20, scale = scale)
I2 = I1

pn1 = LIFPopulation(
        shape = shape1, spike_trace = True, additive_spike_trace = True, tau_s = 10, trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 10, C = 10, threshold = -55
    )
pn1.dt = dt

pn2 = LIFPopulation(
        shape = shape2, spike_trace = True, additive_spike_trace = True, tau_s = 10, trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 10, C = 10, threshold = -55
    )
pn2.dt = dt

con1 = DenseConnection(
        pre = pn1, post = pn2, lr = None, weight_decay = 0.0,
        J = 2, tau_s = 10, trace_scale = 1., dt = dt
    )
con1.dt = dt

con2 = DenseConnection(
        pre = pn2, post = pn1, lr = None, weight_decay = 0.0,
        J = 2, tau_s = 10, trace_scale = 1., dt = dt
    )
con2.dt = dt


monitor1 = Monitor(pn1, state_variables=["s", "u"])
monitor1.set_time_steps(time, dt)
monitor1.reset_state_variables()

monitor2 = Monitor(pn2, state_variables=["s", "u"])
monitor2.set_time_steps(time, dt)
monitor2.reset_state_variables()

I_pop=0
for i in range(len(I1)):
    pn1.forward(I1[i]-I_pop)
    I_pop = con1.compute()
    pn2.forward(I2[i]-I_pop)
    I_pop = con2.compute()
    monitor1.record()
    monitor2.record()

s1 = torch.transpose(monitor1.get("s")*1, 0, 1)
s2 = torch.transpose(monitor2.get("s")*1, 0, 1)

plot = plotting()

plot.plot_population_activity_init(time/scale)
plot.plot_population_activity_update(s1, I1, "p1")
plot.plot_population_activity_update(s2, I2, "p2")
plot.show()


