"""Download MovieLens-1M (https://grouplens.org/datasets/movielens/1m/) into data/ml-1m/.

    python scripts/download_movielens.py [--dir data/ml-1m]

Tries files.grouplens.org first and falls back to a GitHub CSV mirror of the
same 1,000,209 ratings. The data is not committed to the repository.
"""
import argparse

from pmlb.data.movielens import DEFAULT_DIR, download

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()
    print("downloaded from", download(args.dir), "into", args.dir)
