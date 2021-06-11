from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.plotting.plotting import plotting
from cnsproject.utils import step_function, noise_function, step_noise_function, incremental_step_noise_function
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import Connection, randomNormalConnect

import torch
import numpy as np


time = 200
scale = 100
dt = 1
neuron_size = 100
shape_ep1 = (int(neuron_size),)
shape_ep2 = (int(neuron_size),)
shape_ip1 = (int(neuron_size),)

# I_ep1 = incremental_step_noise_function(time = time, I_value = 100, scale = scale, neuron_size = shape_ep1[0], gap = 5)
# I_ep2 = incremental_step_noise_function(time = time, I_value = 100, scale = scale, neuron_size = shape_ep2[0], gap = 3)
# I_ip1 = step_noise_function(time = time, I_value = 100, scale = scale, neuron_size = shape_ip1[0])
I_ep1 = noise_function(time,shape_ep1[0])
I_ep2 = noise_function(time,shape_ep2[0])
I_ip1 = noise_function(time,shape_ip1[0]) * 0

ep1 = LIFPopulation(
        shape = shape_ep1, spike_trace = True, additive_spike_trace = True, tau_s = 10, trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 1, C = 20, threshold = -40, dt = dt
    )
ep2 = LIFPopulation(
        shape = shape_ep2, spike_trace = True, additive_spike_trace = True, tau_s = 10, trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 1, C = 20, threshold = -40, dt = dt
    )
ip1 = LIFPopulation(
        shape = shape_ip1, spike_trace = True, additive_spike_trace = True, tau_s = 10, trace_scale = 1.,
        is_inhibitory = True, learning = False, R = 1, C = 20, threshold = -40, dt = dt
    )

con_ep1_ip1 = Connection(
        pre = ep1, post = ip1, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=20., wstd=5.
    )
con_ep2_ip1 = Connection(
        pre = ep2, post = ip1, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=20., wstd=5.
    )
con_ip1_ep1 = Connection(
        pre = ip1, post = ep1, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=20., wstd=5.
    )
con_ip1_ep2 = Connection(
        pre = ip1, post = ep2, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=20., wstd=5.
    )
con_ep1_ep1 = Connection(
        pre = ep1, post = ep1, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=20., wstd=5.
    )
con_ep2_ep2 = Connection(
        pre = ep2, post = ep2, lr = None, weight_decay = 0.0,
        J = 1, tau_s = 10, trace_scale = 1., dt = dt, connectivity= randomNormalConnect, wmean=20., wstd=5.
    )


monitor_ep1 = Monitor(ep1, state_variables=["s", "u"])
monitor_ep1.set_time_steps(time, dt)
monitor_ep1.reset_state_variables()

monitor_ep2 = Monitor(ep2, state_variables=["s", "u"])
monitor_ep2.set_time_steps(time, dt)
monitor_ep2.reset_state_variables()

monitor_ip1 = Monitor(ip1, state_variables=["s", "u"])
monitor_ip1.set_time_steps(time, dt)
monitor_ip1.reset_state_variables()

out_ep1_ip1 = 0
out_ep2_ip1 = 0
out_ip1_ep1 = 0
out_ip1_ep2 = 0
out_ep1_ep1 = 0
out_ep2_ep2 = 0
for i in range(time):
    ep1.forward(I_ep1[i] - out_ip1_ep1 + out_ep1_ep1)
    ep2.forward(I_ep2[i] - out_ip1_ep2 + out_ep2_ep2)
    ip1.forward(I_ip1[i] + out_ep1_ip1 + out_ep2_ip1)
    out_ep1_ip1 = con_ep1_ip1.compute()
    out_ep2_ip1 = con_ep2_ip1.compute()
    out_ip1_ep1 = con_ip1_ep1.compute()
    out_ip1_ep2 = con_ip1_ep2.compute()
    out_ep1_ep1 = con_ep1_ep1.compute()
    out_ep2_ep2 = con_ep2_ep2.compute()
    monitor_ep1.record()
    monitor_ep2.record()
    monitor_ip1.record()

s_ep1 = torch.transpose(monitor_ep1.get("s")*1, 0, 1)
s_ep2 = torch.transpose(monitor_ep2.get("s")*1, 0, 1)
s_ip1 = torch.transpose(monitor_ip1.get("s")*1, 0, 1)

plot = plotting()

plot.plot_three_population_activity_init(time/scale)
plot.plot_three_population_activity_update(s_ep1, I_ep1, s_ep2, I_ep2, s_ip1, I_ip1, n1="ep1", n2="ep2", n3="ip1")
plot.show()

