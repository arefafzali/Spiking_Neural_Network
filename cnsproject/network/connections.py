"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable

import torch

from cnsproject.network.neural_populations import NeuralPopulation



class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Make sure to implement the `compute`, `update`, and `reset_state_variables`\
    methods in your child class.

    You will need to define the populations you want to connect as `pre` and `post`.\
    In case of learning, you will need to define the learning rate (`lr`) and the \
    learning rule to follow. Attribute `w` is reserved for synaptic weights.\
    However, it has not been predefined or allocated, as it depends on the \
    pattern of connectivity. So make sure to define it in child class initializations \
    appropriately to indicate the pattern of connectivity. The default range of \
    each synaptic weight is [0, 1] but it can be controlled by `wmin` and `wmax`. \
    Synaptic strengths might decay in time and do not last forever. To define \
    the decay rate of the synaptic weights, use `weight_decay` attribute. Also, \
    if you want to control the overall input synaptic strength to each neuron, \
    use `norm` argument to normalize the synaptic weights.

    In case of learning, you have to implement the methods `compute` and `update`. \
    You will use the `compute` method to calculate the activity of post-synaptic \
    population based on the pre-synaptic one. Update of weights based on the \
    learning rule will be implemented in the `update` method. If you find this \
    architecture mind-bugling, try your own architecture and make sure to redefine \
    the learning rule architecture to be compatible with this new architecture \
    of yours.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    lr : float or (float, float), Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float, Optional
        Define rate of decay in synaptic strength. The default is 0.0.

    Keyword Arguments
    -----------------
    learning_rule : LearningRule
        Define the learning rule by which the network will be trained. The\
        default is NoOp (see learning/learning_rules.py for more details).
    wmin : float
        The minimum possible synaptic strength. The default is 0.0.
    wmax : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`,\
        there is no normalization. The default is None.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()

        assert isinstance(pre, NeuralPopulation), \
            "Pre is not a NeuralPopulation instance"
        assert isinstance(post, NeuralPopulation), \
            "Post is not a NeuralPopulation instance"

        self.pre = pre
        self.post = post

        self.weight_decay = weight_decay

        from cnsproject.learning.learning_rules import NoOp
        learning_rule = kwargs.get('learning_rule', NoOp)

        self.learning_rule = learning_rule(
            connection=self,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.wmin = kwargs.get('wmin', 0.)
        self.wmax = kwargs.get('wmax', 1.)
        self.gamma = kwargs.get('gamma', .1)
        self.beta = kwargs.get('beta', 1.)
        self.norm = kwargs.get('norm', None)

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Parameters
        ----------
        s : torch.Tensor
            The pre-synaptic spikes tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule and weight update.

        Keyword Arguments
        -----------------
        learning : bool
            Whether learning is enabled or not. The default is True.
        mask : torch.ByteTensor
            Define a mask to determine which weights to clamp to zero.

        Returns
        -------
        None

        """
        learning = kwargs.get("learning", True)

        if learning:
            self.learning_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass


def fullyConnect(pre, post, J, **kwargs):
    connectivity_matrix = torch.ones(*pre.shape, *post.shape)
    w = torch.ones(*connectivity_matrix.shape)*J/sum(pre.shape)
    return connectivity_matrix, w


def fullyNormalConnect(pre, post, J, **kwargs):
    connectivity_matrix = torch.ones(*pre.shape, *post.shape)
    wmean = kwargs.get('wmean',1.)
    wstd = kwargs.get('wstd',.1)
    w = torch.normal(wmean, wstd, connectivity_matrix.shape)*J/connectivity_matrix.sum(axis=0)
    return connectivity_matrix, w


def randomUniformConnect(pre, post, J, **kwargs):
    connectivity_matrix = torch.rand(*pre.shape, *post.shape)
    wmin = kwargs.get('wmin',0)
    wmax = kwargs.get('wmax',1)
    w = (torch.ones(*connectivity_matrix.shape)*(wmax-wmin)+wmin)*J/connectivity_matrix.sum(axis=0)
    return connectivity_matrix, w


def randomNormalConnect(pre, post, J, **kwargs):
    connectivity_matrix = torch.rand(*pre.shape, *post.shape)
    wmean = kwargs.get('wmean',1.)
    wstd = kwargs.get('wstd',.1)
    w = torch.normal(wmean, wstd, connectivity_matrix.shape)*J/connectivity_matrix.sum(axis=0)
    return connectivity_matrix, w



class Connection(AbstractConnection):
    """
    Specify a synapse between neural populations.

    Implement the connection pattern following the abstract connection\
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        J: float = 0.0,
        tau_s: Union[float, torch.Tensor] = 15.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        dt: int = 1,
        connectivity: Callable = fullyConnect,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        self.J = J
        self.connectivity_matrix, self.w = connectivity(self.pre, self.post, self.J, **kwargs)
        self.traces = torch.zeros(*self.connectivity_matrix.shape)
        self.tau_s = torch.tensor(tau_s)
        self.trace_scale = torch.tensor(trace_scale)
        self.dt = torch.tensor(dt)

    def compute(self) -> torch.tensor:
        self.trace_decay = torch.exp(-self.dt/self.tau_s)
        self.traces *= self.trace_decay
        self.traces += self.trace_scale * self.pre.s.float().reshape(*self.pre.shape, *[1 for i in self.post.shape])
        I = self.w * self.traces * self.connectivity_matrix
        return I.sum(axis=[i for i in range(len(self.pre.shape))])

    def update(self, **kwargs) -> None:
        """
        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.
        """
        self.compute_learning_rate()
        super().update(lr=self.lr, time=kwargs.get("time", None))
        self.w[self.w>self.wmax] = self.wmax
        self.w[self.w<self.wmin] = self.wmin
        return

    def compute_learning_rate(self) -> None:
        self.lr = self.gamma * (((self.wmax - self.w)*(self.w-self.wmin)) ** self.beta)
        return

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class RandomConnection(AbstractConnection):
    """
    Specify a random synaptic connection between neural populations.

    Implement the random connection pattern following the abstract connection\
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class ConvolutionalConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    Implement the convolutional connection pattern following the abstract\
    connection template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class PoolingConnection(AbstractConnection):
    """
    Specify a pooling synaptic connection between neural populations.

    Implement the pooling connection pattern following the abstract connection\
    template. Consider a parameter for defining the type of pooling.

    Note: The pooling operation does not support learning. You might need to\
    make some modifications in the defined structure of this class.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.

        Note: You should be careful with this method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass
