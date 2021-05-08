"""
Module for encoding data into spike.
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterable
from torch.distributions.normal import Normal

import numpy as np
import torch


class AbstractEncoder(ABC):
    """
    Abstract class to define encoding mechanism.

    You will define the time duration into which you want to encode the data \
    as `time` and define the time resolution as `dt`. All computations will be \
    performed on the CPU by default. To handle computation on both GPU and CPU, \
    make sure to set the device as defined in `device` attribute to all your \
    tensors. You can add any other attributes to the child classes, if needed.

    The computation procedure should be implemented in the `__call__` method. \
    Data will be passed to this method as a tensor for further computations. You \
    might need to define more parameters for this method. The `__call__`  should return \
    the tensor of spikes with the shape (time_steps, \*population.shape).

    Arguments
    ---------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        self.time = time
        self.dt = dt
        self.device = device

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> None:
        """
        Compute the encoded tensor of the given data.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.

        Returns
        -------
        None
            It should return the encoded tensor.

        """
        pass


class Time2FirstSpikeEncoder(AbstractEncoder):
    """
    Time-to-First-Spike coding.

    Implement Time-to-First-Spike coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        min_range: int = 0,
        max_range: int = 255,
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        """
        TODO.

        Add other attributes if needed and fill the body accordingly.
        """
        self.min_range = min_range
        self.max_range = max_range



    def __call__(self, data: torch.Tensor) -> torch.tensor:
        """
        TODO.

        Implement the computation for coding the data. Return resulting tensor.
        """
        I = torch.zeros((self.time,*data.shape))
        scaled_data = np.interp(data, (self.min_range, self.max_range), (0, self.time-1))
        scaled_data = np.around(scaled_data)
        scaled_data = torch.from_numpy(scaled_data)

        for i in range(self.time):
            I[i] = (scaled_data == self.time-i)

        return I


    def decode(self, s: np.array, shape: Iterable[int]):
        x = np.sum(s, axis = 0)
        x = x.reshape(shape)
        return x



class PositionEncoder(AbstractEncoder):
    """
    Position coding.

    Implement Position coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        neuron_numbers: int = 1,
        neuron_range_mean: torch.tensor = torch.zeros(0),
        neuron_range_std: torch.tensor = torch.zeros(0),
        min_range: int = 0,
        max_range: int = 255,
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        """
        TODO.

        Add other attributes if needed and fill the body accordingly.
        """
        self.neuron_numbers = neuron_numbers
        self.neuron_range_mean = neuron_range_mean
        self.neuron_range_std = neuron_range_std
        self.min_range = min_range
        self.max_range = max_range
        self.norm_dist = Normal(torch.tensor(self.neuron_range_mean), torch.tensor(self.neuron_range_std))


    def __call__(self, data: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation for coding the data. Return resulting tensor.
        """
        scaled_data = np.interp(data, (self.min_range, self.max_range), (0, self.neuron_numbers))
        scaled_data = torch.tensor(scaled_data)
        scaled_data = scaled_data.reshape(*scaled_data.shape, 1)

        mapped_data = torch.exp(self.norm_dist.log_prob(scaled_data)) / torch.exp(self.norm_dist.log_prob(scaled_data)).mean()

        I = torch.zeros((self.time,*mapped_data.shape))
        scaled_data = np.interp(mapped_data, (self.min_range, self.neuron_numbers), (0, self.time-1))
        scaled_data = np.around(scaled_data)
        scaled_data = torch.from_numpy(scaled_data)

        for i in range(self.time):
            I[i] = (scaled_data == self.time-i)

        return I




class PoissonEncoder(AbstractEncoder):
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        min_range: int = 0,
        max_range: int = 255,
        r: int = 1,
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        """
        TODO.

        Add other attributes if needed and fill the body accordingly.
        """
        self.min_range = min_range
        self.max_range = max_range
        self.max_norm = r*self.dt/self.time

    def __call__(self, data: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation for coding the data. Return resulting tensor.
        """
        I = torch.zeros((self.time,*data.shape))
        norm_data = np.interp(data, (self.min_range, self.max_range), (0, self.max_norm))

        for i in range(self.time):
            I[i] = torch.bernoulli(torch.from_numpy(norm_data))
        
        return I


    def decode(self, s: np.array, shape: Iterable[int]):
        x = np.sum(s, axis = 0)
        x = x.reshape(shape)
        return x

