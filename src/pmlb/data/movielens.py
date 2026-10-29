"""MovieLens-1M: download, load and turn into a linear contextual bandit.

The bandit is semi-synthetic, the standard way to evaluate bandits offline
on rating data:

* reward = "the user likes the movie" (rating >= 4), a Bernoulli reward;
* user features z (30): one-hot gender (2), age band (7) and occupation (21),
  so every protected group owns its own feature coordinates and a small
  group (e.g. the 17 farmers) is a rare feature direction;
* item features y (6): a bias, the movie's overall like rate, and a rank-4
  SVD embedding of the centred user x movie like matrix;
* the true parameter theta* (30 x 6 = 180) is the ridge fit of
  like ~ kron(z_u, y_i) on every observed rating of the n_items most rated
  movies, so the environment is exactly linear and its expected rewards
  are the fitted like probabilities (all inside (0, 1) for the default
  settings);
* each round a user is drawn uniformly from the 6,040 users (so groups
  arrive in proportion to their size) and K candidate movies uniformly
  from the item pool.
"""
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from ..envs import BERNOULLI
from ..linear.envs import LinearEnv

OFFICIAL_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# Plain-CSV copy of the same files, used when grouplens.org is unreachable.
MIRROR_URL = "https://raw.githubusercontent.com/khanhnamle1994/movielens/master/{}.csv"
DEFAULT_DIR = os.path.join("data", "ml-1m")
N_RATINGS = 1_000_209

AGE_BANDS = [1, 18, 25, 35, 45, 50, 56]
AGE_NAMES = ["<18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
OCCUPATIONS = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student",
               "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker",
               "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist",
               "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]


def _from_official(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))

    def dat(name, cols):
        text = z.read(f"ml-1m/{name}.dat").decode("latin-1")
        return pd.read_csv(io.StringIO(text), sep="::", engine="python", names=cols)

    ratings = dat("ratings", ["user_id", "movie_id", "rating", "timestamp"])
    users = dat("users", ["user_id", "gender", "age", "occupation", "zip"])
    movies = dat("movies", ["movie_id", "title", "genres"])
    return ratings, users, movies


def _from_mirror():
    def get(name):
        with urllib.request.urlopen(MIRROR_URL.format(name), timeout=120) as r:
            return pd.read_csv(io.BytesIO(r.read()), sep="\t", index_col=0, encoding="latin-1")

    ratings = get("ratings")[["user_id", "movie_id", "rating", "timestamp"]]
    users = get("users")[["user_id", "gender", "age", "occupation", "zipcode"]].rename(columns={"zipcode": "zip"})
    movies = get("movies")[["movie_id", "title", "genres"]]
    return ratings, users, movies


def download(data_dir=DEFAULT_DIR):
    """Fetch MovieLens-1M into data_dir as ratings.csv, users.csv, movies.csv."""
    os.makedirs(data_dir, exist_ok=True)
    try:
        with urllib.request.urlopen(OFFICIAL_URL, timeout=120) as r:
            ratings, users, movies = _from_official(r.read())
        source = OFFICIAL_URL
    except Exception as e:  # noqa: BLE001
        print(f"grouplens.org failed ({e}); using the GitHub mirror")
        ratings, users, movies = _from_mirror()
        source = MIRROR_URL.format("*")
    assert len(ratings) == N_RATINGS and len(users) == 6040, "unexpected MovieLens-1M size"
    ratings.to_csv(os.path.join(data_dir, "ratings.csv"), index=False)
    users.to_csv(os.path.join(data_dir, "users.csv"), index=False)
    movies.to_csv(os.path.join(data_dir, "movies.csv"), index=False)
    return source


def load(data_dir=DEFAULT_DIR):
    if not os.path.exists(os.path.join(data_dir, "ratings.csv")):
        raise FileNotFoundError(f"{data_dir} is empty; run `python scripts/download_movielens.py` first")
    return tuple(pd.read_csv(os.path.join(data_dir, f"{n}.csv")) for n in ("ratings", "users", "movies"))


def user_features(users):
    """One-hot gender, age band, occupation; and each user's three group ids."""
    n = len(users)
    g = (users.gender.values == "F").astype(np.int64)
    a = users.age.map({b: k for k, b in enumerate(AGE_BANDS)}).values.astype(np.int64)
    o = users.occupation.values.astype(np.int64)
    ugroups = np.stack([g, 2 + a, 9 + o], axis=1)
    Z = np.zeros((n, 30))
    Z[np.arange(n)[:, None], ugroups] = 1.0
    names = ["male", "female"] + [f"age {s}" for s in AGE_NAMES] + OCCUPATIONS
    attrs = ["gender"] * 2 + ["age"] * 7 + ["occupation"] * 21
    return Z, ugroups, names, attrs


def fit_model(ratings, users, n_items=300, rank=4, ridge=1.0):
    """Item features and the ridge fit of theta*. Returns (Z, Y, theta, ugroups, names, attrs, items)."""
    items = ratings.movie_id.value_counts().index[:n_items].to_numpy()
    r = ratings[ratings.movie_id.isin(items)]
    uid = pd.Series(np.arange(len(users)), index=users.user_id.values)
    iid = pd.Series(np.arange(n_items), index=items)
    ui, ii = uid[r.user_id].values, iid[r.movie_id].values
    like = (r.rating.values >= 4).astype(np.float64)

    like_rate = np.bincount(ii, weights=like, minlength=n_items) / np.bincount(ii, minlength=n_items)
    M = np.zeros((len(users), n_items))
    M[ui, ii] = like - like_rate[ii]
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    emb = Vt[:rank].T * np.sqrt(s[:rank] / s[0])
    Y = np.hstack([np.ones((n_items, 1)), like_rate[:, None], emb])

    Z, ugroups, names, attrs = user_features(users)
    dz, dy = Z.shape[1], Y.shape[1]
    A, b = np.zeros((dz * dy, dz * dy)), np.zeros(dz * dy)
    for c in np.array_split(np.arange(like.size), 20):
        F = (Z[ui[c]][:, :, None] * Y[ii[c]][:, None, :]).reshape(c.size, -1)
        A += F.T @ F
        b += F.T @ like[c]
    theta = np.linalg.solve(A + ridge * np.eye(dz * dy), b)
    return Z, Y, theta, ugroups, names, attrs, items


def make_env(T, K=20, n_items=300, rank=4, seed=0, data_dir=DEFAULT_DIR):
    ratings, users, _ = load(data_dir)
    Z, Y, theta, ugroups, names, attrs, _ = fit_model(ratings, users, n_items, rank)
    rng = np.random.default_rng(seed)
    stream_users = rng.integers(0, Z.shape[0], T)
    cands = np.argsort(rng.random((T, n_items)), axis=1)[:, :K]
    env = LinearEnv(Z=Z, Y=Y, theta=theta, users=stream_users, cands=cands, ugroups=ugroups,
                    G=len(names), sigma2=0.25, reward_type=BERNOULLI, group_names=names, group_attr=attrs)
    P = env.all_means()
    assert P.min() > 0 and P.max() < 1, "fitted like probabilities must lie in (0, 1)"
    return env
