from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.plotting.plotting import plotting
from cnsproject.utils import step_function, two_way_step_function, random_step_function
from cnsproject.network.monitors import Monitor

import torch
import numpy as np



def single_neuron_time(
        time, dt, scale, step_size, I_function, shape, spike_trace,
        additive_spike_trace, tau_s, trace_scale,
        is_inhibitory, learning, R, C, threshold = -55
    ):
    
    I = I_function(time, step_size, scale)
    neuron = LIFPopulation(
            shape, spike_trace, additive_spike_trace, tau_s, trace_scale,
            is_inhibitory, learning, R, C, threshold
        )
    neuron.dt = dt
    monitor = Monitor(neuron, state_variables=["s", "u"])
    monitor.set_time_steps(time, dt)
    monitor.reset_state_variables()
    for i in range(len(I)):
        neuron.forward(I[i][0])
        monitor.record()
    return neuron, I, torch.transpose(monitor.get("s")*1, 0, 1), monitor.get("u")


time = 1500
scale = 100

plot = plotting()

neuron, I, s, u = single_neuron_time(
        time = time, dt = 1, scale = scale, step_size = 1,
        I_function = step_function, shape = (1,), spike_trace = True,
        additive_spike_trace = True, tau_s = 10., trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 10, C = 10
    )

plot.plot_ut_it_init(time/scale)
plot.plot_ut_it_update(I, u, neuron.threshold, s[0].nonzero(as_tuple=True)[0], "default")
plot.show()


plot.plot_fi_init()
spikes = []
for x in range(15):
    _, _, s, _ = single_neuron_time(
        time = time, dt = 1, scale = scale, step_size = x,
        I_function = step_function, shape = (1,), spike_trace = True,
        additive_spike_trace = True, tau_s = 10., trace_scale = 1.,
        is_inhibitory = False, learning = False, R = 10, C = 10
    )
    spikes.append(s[0].sum())
    plot.plot_fi_update(spikes)

plot.show()




# spike_mom = np.array([0])
# spike_freq = np.array([0])
# spike_mom = np.append(spike_mom, s.nonzero()[0])
# spike_freq = np.append(spike_freq, np.cumsum(1/s.nonzero()[0]))
