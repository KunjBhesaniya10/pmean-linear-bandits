"""Run all presented bandit experiments and generate publication figures."""
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from experiments import mab_reproduction, linear_synthetic, movielens


def main():
    start_total = time.time()

    print("=" * 70)
    print("STAGE 1: Running MAB Reproduction Experiments (a)-(f)...")
    print("=" * 70)
    t0 = time.time()
    instances = mab_reproduction.make_instances()
    specs = mab_reproduction.experiment_specs(instances)
    all_rows = []
    for expt, spec in specs.items():
        all_rows.extend(mab_reproduction.run_experiment(expt, spec, force=False))
    mab_reproduction.write_summary(all_rows, specs)
    print(f"--> Done with MAB Reproduction in {time.time() - t0:.1f}s\n")

    print("=" * 70)
    print("STAGE 2: Running Linear Synthetic Experiment 1 (Benign arms)...")
    print("=" * 70)
    t0 = time.time()
    linear_synthetic.expt1(force=False)
    print(f"--> Done with Linear Synthetic Expt 1 in {time.time() - t0:.1f}s\n")

    print("=" * 70)
    print("STAGE 3: Running Linear Synthetic Experiment 2 (Rare niche arms)...")
    print("=" * 70)
    t0 = time.time()
    linear_synthetic.expt2(force=False)
    print(f"--> Done with Linear Synthetic Expt 2 in {time.time() - t0:.1f}s\n")

    print("=" * 70)
    print("STAGE 4: Running MovieLens-1M Streaming Benchmark...")
    print("=" * 70)
    t0 = time.time()
    movielens_data = os.path.join("data", "ml-1m", "ratings.csv")
    if os.path.exists(movielens_data):
        sys.argv = ["movielens"]
        movielens.main()
        print(f"--> Done with MovieLens-1M in {time.time() - t0:.1f}s\n")
    else:
        print("MovieLens-1M data not detected in data/ml-1m/.")
        print("Run `python scripts/download_movielens.py` to download the dataset before running Stage 4.\n")

    print("=" * 70)
    print(f"ALL EXPERIMENTS COMPLETED in {time.time() - start_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
