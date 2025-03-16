from typing import Any

import gymnasium.spaces
import numpy as np
from gymnasium.spaces.space import MaskNDArray
from numpy._typing import NDArray


class MultiDiscreteUnique(gymnasium.spaces.MultiDiscrete):
    """
    Wraps Gymnasium's MultiDiscrete space
     to guarantee that all values in a single sample are unique
     Can be used with multidimensional spaces but really shouldn't be,
     given performance concerns.
    """
    def sample(
        self,
        mask: tuple[MaskNDArray, ...] | None = None,
        probability: tuple[MaskNDArray, ...] | None = None,
    ) -> NDArray[np.integer[Any]]:
        result = super().sample(mask=mask, probability=probability, )
        while len(set(result.flatten())) != len(result.flatten()):
            result = super().sample(mask=mask, probability=probability)
        return result

    def contains(self, x: Any) -> bool:
        arr = np.array(x)
        is_unique = len(set(arr.flatten())) == len(arr.flatten())
        return is_unique and super().contains(x)