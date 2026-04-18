from abc import ABC, abstractmethod

from numpy.typing import NDArray


class ControllerBackend(ABC):
    """Backend interface consumed by the ROS frontend."""

    @abstractmethod
    def get_action(self, observation: NDArray) -> NDArray:
        """Return control action [v, omega] for the provided state."""
        raise NotImplementedError
