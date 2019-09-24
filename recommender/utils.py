from typing import Tuple, Callable

from cached_property import cached_property
import pandas as pd
import os

from sklearn.model_selection import train_test_split


def download_and_extract(url, outputdir):
    import requests, zipfile, io

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(outputdir)


class MovieLens100K:
    def __init__(self, path):
        self._path = os.path.join(path, "ml-100k")

    def _load_or_download_first(
        self, loader: Callable[[None], pd.DataFrame]
    ) -> pd.DataFrame:
        try:
            return loader()
        except:
            download_and_extract(
                "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                os.path.join(self._path, ".."),
            )
        return loader()

    @cached_property
    def ratings(self) -> pd.DataFrame:
        loader = lambda: pd.read_csv(
            os.path.join(self._path, "u.data"),
            sep="\t",
            header=None,
            names=["userId", "movieId", "rating", "timestamp"],
            index_col=False,
            dtype={"userId": "category", "movieId": "category"},
        )

        return self._load_or_download_first(loader)

    @cached_property
    def movies(self) -> pd.DataFrame:
        loader = lambda: pd.read_csv(
            os.path.join(self._path, "u.item"),
            sep="|",
            header=None,
            names=["id", "title"],
            usecols=["id", "title"],
            index_col="id",
            encoding='ISO-8859-1'
        )

        return self._load_or_download_first(loader)
