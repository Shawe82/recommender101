from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow import Tensor
from tensorflow.keras.layers import (
    Embedding,
    Input,
    Dot,
    Lambda,
    Activation,
    Add,
    Dense,
    Concatenate,
    Dropout,
)
from tensorflow.keras.models import Model

from recommender.utils import MovieLens100K


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


def recommender_model(
    n_users: int,
    n_movies: int,
    emb_dim: int,
    min_max: Optional[Tuple[float, float]] = None,
    use_bias: bool = False,
) -> Model:
    users = Input(shape=(1,))
    u_emb = Embedding(input_dim=n_users, output_dim=emb_dim)(users)

    movies = Input(shape=(1,))
    m_emb = Embedding(input_dim=n_movies, output_dim=emb_dim)(movies)

    ratings = Dot(axes=2)([u_emb, m_emb])

    if use_bias:
        u_bias = Embedding(input_dim=n_users, output_dim=1)(users)
        m_bias = Embedding(input_dim=n_movies, output_dim=1)(movies)
        ratings = Add()([ratings, u_bias, m_bias])

    ratings = SqueezeLayer(axis=1)(ratings)
    ratings = ratings if min_max is None else ScalingLayer(*min_max)(ratings)
    return Model(inputs=[users, movies], outputs=ratings)


def recommender_model_hyb(
    n_users: int,
    n_movies: int,
    e_user_dim: int,
    e_movies_dim: int,
    min_max: Optional[Tuple[float, float]] = None,
) -> Model:
    users = Input(shape=(1,))
    u_emb = Embedding(input_dim=n_users, output_dim=e_user_dim)(users)

    movies = Input(shape=(1,))
    m_emb = Embedding(input_dim=n_movies, output_dim=e_movies_dim)(movies)
    rep = Concatenate()([u_emb, m_emb])
    rep = Dropout(0.2)(rep)
    rep = Dense(50, activation="relu")(rep)
    rep = Dropout(0.2)(rep)
    ratings = Dense(1, activation=None)(rep)

    ratings = (
        ratings
        if min_max is None
        else ScalingLayer(min_val=min_max[0], max_val=min_max[1])(ratings)
    )
    return Model(inputs=[users, movies], outputs=ratings)


def optimal_emb_dim(n_categories) -> int:
    return int(min(600, round(1.6 * n_categories ** 0.56)))


if __name__ == "__main__":
    data = MovieLens100K("../data/")
    x_train, x_test, y_train, y_test = train_test_split(
        np.array([data.ratings.userId.cat.codes, data.ratings.movieId.cat.codes]).T,
        data.ratings.rating,
        train_size=0.8,
        random_state=42,
    )

    n_users = data.ratings.userId.nunique()
    n_movies = data.ratings.movieId.nunique()
    emb_dim = 50  # max(optimal_emb_dim(n_users),optimal_emb_dim(n_movies))
    min_max = (data.ratings.rating.min(), data.ratings.rating.max())
    mdl = recommender_model(n_users, n_movies, emb_dim, min_max, use_bias=True)

    mdl.compile("adam", loss="mse")
    print(mdl.summary())

    history = mdl.fit(
        np.split(x_train, 2, axis=1),
        y_train,
        batch_size=64,
        epochs=5,
        verbose=1,
        validation_data=(np.split(x_test, 2, axis=1), y_test),
    )

    # take random user
    movies = data.ratings.movieId.unique()
    user = data.ratings.userId.sample(1)
    user_rows = data.ratings[data.ratings.userId == user.values]
    # Take all the movies the user has not watched yet, as we don't want to recommend already seen stuff
    not_yet_watched_movies = movies[movies.isin(user_rows.movieId)]

    estimated_ratings = mdl.predict(
        [
            np.full_like(not_yet_watched_movies.codes, user.cat.codes, dtype=int),
            not_yet_watched_movies.codes,
        ]
    ).flatten()
    # Take the 10 best ratings from the movies the user has not yet watched
    # We go back from categories to integers in order to find the movies
    recommends = not_yet_watched_movies[estimated_ratings.argsort()[-10:]].astype(int)
    print("What we suggest")
    print(data.movies.loc[recommends])

    print("What the user liked most")
    previous = (
        user_rows.sort_values(by="rating", ascending=False)
        .movieId[:10]
        .astype("int")
        .values
    )
    print(data.movies.loc[previous])