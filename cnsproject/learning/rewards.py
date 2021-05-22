"""
Module for reward dynamics.

TODO.

Define your reward functions here.
"""

from abc import ABC, abstractmethod
from typing import Callable


class AbstractReward(ABC):
    """
    Abstract class to define reward function.

    Make sure to implement the abstract methods in your child class.

    To implement your dopamine functionality, You will write a class \
    inheriting this abstract class. You can add attributes to your \
    child class. The dynamics of dopamine function (DA) will be \
    implemented in `compute` method. So you will call `compute` in \
    your reward-modulated learning rules to retrieve the dopamine \
    value in the desired time step. To reset or update the defined \
    attributes in your reward function, use `update` method and \
    remember to call it your learning rule computations in the \
    right place.
    """

    @abstractmethod
    def __init__(
        self,
        **kwargs
    ) -> None:
        pass

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass

class Reward(AbstractReward):

    def __init__(
        self,
        tau_d: float = 1.,
        **kwargs
    ) -> None:
        super().__init__(
            **kwargs
        )
        self.dopamine = 0
        self.tau_d = tau_d
        self.dt = kwargs.get("dt", 1)
        return

    def compute(self, **kwargs) -> int:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """
        reward = kwargs.get("reward", 0)
        self.dopamine += self.dt * (-self.dopamine/self.tau_d + reward)
        return self.dopamine

    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass

