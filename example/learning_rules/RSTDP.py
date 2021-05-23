import torch
from PIL import Image
import numpy as np

from cnsproject.plotting.plotting import plotting
from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.encoding.encoders import PoissonEncoder
from cnsproject.network.connections import Connection, fullyNormalConnect
from cnsproject.learning.learning_rules import RSTDP
from cnsproject.learning.rewards import Reward


# 0 -> im1 & 1 -> im2
def DA(time, n0, n1) -> int:
    r = 0
    time = time/20
    if n0 < n1:
        if ((time < 5) or ((time<15) and (time>10)) or ((time<25) and (time>20))):
            r = -1
        else:
            r = 1
    # elif n0 > n1:
    else:
        if ((time < 5) or ((time<15) and (time>10)) or ((time<25) and (time>20))):
            r = 1
        else:
            r = -1
    return r


time = 600
scale = 100
dt = 1
shape1 = (20,20)
shape2 = (1,)
shape3 = (1,)

im1 = Image.open("./example/learning_rules/img1.png").convert('L')
im1 = im1.resize(shape1, Image.ANTIALIAS)
im2 = Image.open("./example/learning_rules/img.jpg").convert('L')
im2 = im2.resize(shape1, Image.ANTIALIAS)
s1 = torch.from_numpy(np.asarray(im1).copy())
s2 = torch.from_numpy(np.asarray(im2).copy())

encoder = PoissonEncoder(time = int(time/6), r=10)
I1 = encoder(s1)
I2 = encoder(s2)

I = torch.cat((I1,I2,I1,I2,I1,I2), 0)


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


con1 = Connection(
        pre = n1, post = n2, lr = None, weight_decay = 0.05,
        J = 1, tau_s = 5, trace_scale = 10., dt = dt, connectivity = fullyNormalConnect, wmean=5., wstd=.5,
        learning_rule=RSTDP, beta=1., gamma=.2
    )

con2 = Connection(
        pre = n1, post = n3, lr = None, weight_decay = 0.05,
        J = 1, tau_s = 5, trace_scale = 10., dt = dt, connectivity = fullyNormalConnect, wmean=5., wstd=.5,
        learning_rule=RSTDP, beta=1., gamma=.2
    )


monitor1 = Monitor(n1, state_variables=["s"])
monitor1.set_time_steps(time, dt)
monitor1.reset_state_variables()

monitor2 = Monitor(n2, state_variables=["s"])
monitor2.set_time_steps(time, dt)
monitor2.reset_state_variables()

monitor3 = Monitor(n3, state_variables=["s"])
monitor3.set_time_steps(time, dt)
monitor3.reset_state_variables()

reward = Reward()

I_ex1 = 0
I_ex2 = 0
for i in range(len(I)):
    n1.forward(I[i])
    n2.forward(I_ex1)
    n3.forward(I_ex2)
    I_ex1 = con1.compute()
    I_ex2 = con2.compute()
    r = DA(i, n2.s.sum(), n3.s.sum())
    d = reward.compute(reward=r)
    con1.update(learning=True,dopamine=d)
    con2.update(learning=True,dopamine=d)
    monitor1.record()
    monitor2.record()
    monitor3.record()


s1 = monitor1.get("s").flatten(start_dim=1)
s2 = monitor2.get("s").flatten(start_dim=1)
s3 = monitor3.get("s").flatten(start_dim=1)
as2 = s2.sum(axis=1)
as3 = s3.sum(axis=1)

plot = plotting()

plot.plot_Rlearning_init(time/scale)
plot.plot_Rlearning_update(s1.T, s2.T, as2, mode="2")
plot.plot_Rlearning_update(s1.T, s3.T, as3,start_idx=sum(list(shape2)), mode="3")
plot.show()

