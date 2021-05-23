"""
Module for learning rules.
"""

from abc import ABC
from typing import Callable, Union, Optional, Sequence

import numpy as np
import torch

from cnsproject.network.connections import AbstractConnection
# from cnsproject.learning.rewards import Reward


class LearningRule(ABC):
    """
    Abstract class for defining learning rules.

    Each learning rule will be applied on a synaptic connection defined as \
    `connection` attribute. It possesses learning rate `lr` and weight \
    decay rate `weight_decay`. You might need to define more parameters/\
    attributes to the child classes.

    Implement the dynamics in `update` method of the classes. Computations \
    for weight decay and clamping the weights has been implemented in the \
    parent class `update` method. So do not invent the wheel again and call \
    it at the end  of the child method.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        if lr is None:
            lr = [0., 0.]
        elif isinstance(lr, float) or isinstance(lr, int):
            lr = [lr, lr]

        self.lr = torch.tensor(lr, dtype=torch.float)

        self.connection = connection
        self.weight_decay = weight_decay

    def update(self) -> None:
        """
        Abstract method for a learning rule update.

        Returns
        -------
        None

        """
        if self.weight_decay:
            self.connection.w *= self.weight_decay

        if (
            self.connection.wmin != -np.inf or self.connection.wmax != np.inf
        ) and not isinstance(self.connection, NoOp):
            self.connection.w.clamp_(self.connection.wmin,
                                     self.connection.wmax)


class NoOp(LearningRule):
    """
    Learning rule with no effect.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Only take care about synaptic decay and possible range of synaptic
        weights.

        Returns
        -------
        None

        """
        super().update()


class STDP(LearningRule):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        self.lr = kwargs.get("lr", .1)
        pre_trace = self.connection.pre.traces.reshape((*self.connection.pre.shape,*[1]*len(self.connection.post.shape)))
        post_trace = self.connection.post.traces.reshape((*[1]*len(self.connection.pre.shape),*self.connection.post.shape))
        self.connection.w += self.connection.dt * (
            self.lr * pre_trace * self.connection.post.s.reshape((*[1]*len(self.connection.pre.shape),*self.connection.post.shape))
             - self.lr * post_trace * self.connection.pre.s.reshape((*self.connection.pre.shape,*[1]*len(self.connection.post.shape)))
        ) - self.weight_decay * self.connection.w
        return


class FlatSTDP(LearningRule):
    """
    Flattened Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.trace_periode = kwargs.get("trace_periode", 50)
        self.pre_traces = torch.zeros((self.trace_periode, *self.connection.pre.shape))
        self.post_traces = torch.zeros((self.trace_periode, *self.connection.post.shape))
        """
        Consider the additional required parameters and fill the body\
        accordingly.
        """
        return

    def update(self, **kwargs) -> None:
        """
        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        self.lr = kwargs.get("lr", .1)

        self.pre_traces = torch.cat((self.pre_traces[1:], self.connection.pre.s.reshape((1,*self.connection.pre.s.shape))), 0)
        self.post_traces = torch.cat((self.post_traces[1:], self.connection.post.s.reshape((1,*self.connection.post.s.shape))), 0)

        pre_trace = self.pre_traces.sum(axis=0)
        post_trace = self.post_traces.sum(axis=0)

        pre_trace = pre_trace.reshape((*self.connection.pre.shape,*[1]*len(self.connection.post.shape)))
        post_trace = post_trace.reshape((*[1]*len(self.connection.pre.shape),*self.connection.post.shape))

        self.connection.w += self.connection.dt * (
            self.lr * pre_trace * self.connection.post.s.reshape((*[1]*len(self.connection.pre.shape),*self.connection.post.shape))
             - self.lr * post_trace * self.connection.pre.s.reshape((*self.connection.pre.shape,*[1]*len(self.connection.post.shape)))
        ) - self.weight_decay * self.connection.w

        self.connection.w = torch.nan_to_num(self.connection.w, nan=0)

        return

class RSTDP(LearningRule):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        DA: Callable = None,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        Consider the additional required parameters and fill the body\
        accordingly.
        """
        self.tau_c = kwargs.get("tau_c", 1.)
        # self.reward = Reward()
        # self.DA = DA
        self.c = torch.zeros((*self.connection.pre.shape, *self.connection.post.shape))
        # self.c = torch.zeros(*self.connection.post.shape)

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        self.lr = kwargs.get("lr", .1)
        dopamine = kwargs.get("dopamine", None)
        pre_trace = self.connection.pre.traces.reshape((*self.connection.pre.shape,*[1]*len(self.connection.post.shape)))
        post_trace = self.connection.post.traces.reshape((*[1]*len(self.connection.pre.shape),*self.connection.post.shape))
        x = (
                self.lr * pre_trace * self.connection.post.s.reshape(*post_trace.shape)
                - self.lr * post_trace * self.connection.pre.s.reshape(*pre_trace.shape)
            ) - self.weight_decay * self.connection.w

        # print(self.connection.pre.s.type(torch.int))
        self.c += self.connection.dt * (-self.c/self.tau_c + x * torch.bitwise_or(
                self.connection.pre.s.reshape(*pre_trace.shape).type(torch.int), 
                self.connection.post.s.reshape(*post_trace.shape).type(torch.int)
            ))
        # print(self.c)
        # r = self.DA(time, self.connection.pre.s.sum(), self.connection.post.s.sum())
        # # self.c += self.connection.dt * (-self.c/self.tau_c + x * torch.bitwise_or(
        # #     self.connection.post.s[0], self.connection.post.s[1]
        # #     ))
        # # r = self.DA(time, self.connection.post.s[0], self.connection.post.s[1])

        # dopamine = self.reward.compute(reward=r)
        self.connection.w += self.connection.dt * (self.c * dopamine)

        return


class FlatRSTDP(LearningRule):
    """
    Flattened Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        pass
