from dataclasses import dataclass

from tensorflow import Tensor
from tensorflow.python.keras.api._v2.keras import backend as K
from tensorflow.python.keras.layers import Lambda, Activation


@dataclass
class SqueezeLayer:
    axis: int

    def __call__(self, x: Tensor) -> Tensor:
        return Lambda(lambda xx: K.squeeze(xx, axis=self.axis))(x)


@dataclass
class ScalingLayer:
    min_val: float
    max_val: float

    def __call__(self, x: Tensor) -> Tensor:
        return Lambda(lambda in_: in_ * (self.max_val - self.min_val) + self.min_val)(
            Activation("sigmoid")(x)
        )