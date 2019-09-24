from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Embedding,
    Input,
    Dot,
    Add,
)
from tensorflow.keras.models import Model

from recommender.layers import SqueezeLayer, ScalingLayer
from recommender.utils import MovieLens100K

DATA = MovieLens100K("../data/")


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



class Recommender:
    def __init__(self, mdl: Model):
        self._mdl = mdl
        self._movies = pd.Series(dict(enumerate(DATA.ratings.movieId.cat.categories)))

    def __call__(self, user) -> Tuple[pd.DataFrame, np.array]:
        user_rows = DATA.ratings[DATA.ratings.userId == user.values]
        # Take all the movies the user has not watched yet, as we don't want to recommend already seen stuff
        not_yet_watched_movies = self._movies[~self._movies.isin(user_rows.movieId)]
        # Rate the unseeb ones
        estimated_ratings = self._mdl.predict(
            [
                np.full_like(not_yet_watched_movies, user.cat.codes),
                not_yet_watched_movies.index.values,
            ]
        ).flatten()

        # Take the 10 best ratings from the movies the user has not yet watched
        # We go back from categories to integers in order to find the movies
        index = estimated_ratings.argsort()[:-10:-1]
        recommends = not_yet_watched_movies.iloc[index].astype(int).values
        return DATA.movies.loc[recommends], estimated_ratings[index]


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = train_test_split(
        np.array([DATA.ratings.userId.cat.codes, DATA.ratings.movieId.cat.codes]).T,
        DATA.ratings.rating,
        train_size=0.8,
        random_state=42,
    )

    n_users = DATA.ratings.userId.nunique()
    n_movies = DATA.ratings.movieId.nunique()
    emb_dim = 50  # max(optimal_emb_dim(n_users),optimal_emb_dim(n_movies))
    min_max = (DATA.ratings.rating.min(), DATA.ratings.rating.max())
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
    recommender = Recommender(mdl)
    user = DATA.ratings.userId.sample(1)

    print("What we suggest")
    print(recommender(user))
