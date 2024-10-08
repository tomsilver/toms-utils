"""Utilities for gymnasium-like spaces."""

from typing import Any, Sequence, TypeVar

import gymnasium as gym
import numpy as np

_Element = TypeVar("_Element")


class EnumSpace(gym.spaces.Space[_Element]):
    """A space defined by an arbitrary finite list."""

    def __init__(
        self,
        elements: Sequence[_Element],
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(shape=None, dtype=None, seed=seed)
        self._elements = list(elements)
        self._num_elements = len(self._elements)

    def sample(self, mask: Any | None = None) -> _Element:
        return self._elements[self.np_random.choice(self._num_elements)]

    def contains(self, x: Any) -> bool:
        return x in self._elements

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"EnumSpace({self._elements})"
