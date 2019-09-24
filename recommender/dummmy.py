from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow.keras.backend as K
from tensorflow import Tensor
from tensorflow.keras.layers import Embedding, Input, Dot, Lambda, Activation, Add
from tensorflow.keras.models import Model


@dataclass
class SqueezLayer:
    axis: int

    def __call__(self, x: Tensor) -> Tensor:
        return Lambda(lambda xx: K.squeeze(xx, axis=self.axis))(x)


@dataclass
class ScalingLayer:
    min: float
    max: float

    def __call__(self, x: Tensor) -> Tensor:
        return Lambda(
            lambda in_: in_ * (self.max - self.min) + self.min)(
            Activation("sigmoid")(x)
        )


def recommender_model(
    n_users: int,
    n_movies: int,
    emb_dim: int,
    min_max: Optional[Tuple[float, float]] = None,
    use_bias: bool = False,
) -> Model:
    users = Input(shape=(1,))
    u_e = Embedding(input_dim=n_users, output_dim=emb_dim)(users)

    movies = Input(shape=(1,))
    m_e = Embedding(input_dim=n_movies, output_dim=emb_dim)(movies)

    ratings = SqueezLayer(axis=1)(Dot(axes=2)([u_e, m_e]))

    if use_bias:
        u_bias = Embedding(input_dim=n_users, output_dim=1)(users)
        m_bias = Embedding(input_dim=n_movies, output_dim=1)(movies)
        ratings = Add()([ratings, u_bias, m_bias])

    ratings = ratings if min_max is None else ScalingLayer(*min_max)(ratings)
    model = Model(inputs=[users, movies], outputs=ratings)
    model.compile("adam", loss="mse")
    return model
