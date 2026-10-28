| algorithm | p | exploration rounds (mean ± sd) | runs ending on the optimal arm | p-mean regret at t=1e4 | p-mean regret at T |
|---|---|---|---|---|---|
| Explore-then-LinUCB, naive rule | 1 | 5220 ± 151 | 100% | 0.2031 | 0.0104 |
| Explore-then-LinUCB, naive rule | 0 | 5220 ± 151 | 100% | 0.2272 | 0.0132 |
| Explore-then-LinUCB, naive rule | -1 | 5220 ± 151 | 100% | 0.2503 | 0.0170 |
| Explore-then-LinUCB, naive rule | -2 | 19120 ± 209 | 100% | 0.3879 | 0.0742 |
| Explore-then-LinUCB, naive rule | -5 | 112805 ± 587 | 100% | 0.3886 | 0.3271 |
| design only (naive rule) | 1 | 6452 ± 149 | 100% | 0.2607 | 0.0132 |
| design only (naive rule) | 0 | 6452 ± 149 | 100% | 0.2856 | 0.0170 |
| design only (naive rule) | -1 | 6452 ± 149 | 100% | 0.3077 | 0.0224 |
| design only (naive rule) | -2 | 23494 ± 262 | 100% | 0.4078 | 0.0975 |
| design only (naive rule) | -5 | 138687 ± 745 | 100% | 0.4118 | 0.3735 |
| coverage rule only (uniform) | 1 | 200000 ± 0 | 0% | 0.3870 | 0.3871 |
| coverage rule only (uniform) | 0 | 200000 ± 0 | 0% | 0.3873 | 0.3874 |
| coverage rule only (uniform) | -1 | 200000 ± 0 | 0% | 0.3876 | 0.3877 |
| coverage rule only (uniform) | -2 | 200000 ± 0 | 0% | 0.3879 | 0.3879 |
| coverage rule only (uniform) | -5 | 200000 ± 0 | 0% | 0.3887 | 0.3888 |
| Explore-then-LinUCB (ours) | 1 | 15584 ± 201 | 100% | 0.4039 | 0.0315 |
| Explore-then-LinUCB (ours) | 0 | 15584 ± 201 | 100% | 0.4052 | 0.0402 |
| Explore-then-LinUCB (ours) | -1 | 15584 ± 201 | 100% | 0.4065 | 0.0521 |
| Explore-then-LinUCB (ours) | -2 | 56588 ± 350 | 100% | 0.4080 | 0.1953 |
| Explore-then-LinUCB (ours) | -5 | 200000 ± 0 | 0% | 0.4120 | 0.4119 |
| Explore-then-Greedy, naive rule | 1 | 5213 ± 133 | 76% | 0.2147 | 0.0343 |
| Explore-then-Greedy, naive rule | 0 | 5213 ± 133 | 76% | 0.2364 | 0.0368 |
| Explore-then-Greedy, naive rule | -1 | 5213 ± 133 | 76% | 0.2572 | 0.0401 |
| Explore-then-Greedy, naive rule | -2 | 19155 ± 178 | 96% | 0.3879 | 0.0772 |
| Explore-then-Greedy, naive rule | -5 | 112814 ± 581 | 100% | 0.3889 | 0.3270 |
| Explore-then-Greedy (ours) | 1 | 15525 ± 203 | 100% | 0.4039 | 0.0314 |
| Explore-then-Greedy (ours) | 0 | 15525 ± 203 | 100% | 0.4052 | 0.0401 |
| Explore-then-Greedy (ours) | -1 | 15525 ± 203 | 100% | 0.4065 | 0.0519 |
| Explore-then-Greedy (ours) | -2 | 56618 ± 328 | 100% | 0.4077 | 0.1953 |
| Explore-then-Greedy (ours) | -5 | 200000 ± 0 | 0% | 0.4122 | 0.4120 |
| LinNash (Sawarni et al.) | 1 | 14020 ± 0 | 0% | 0.4316 | 0.3317 |
| LinNash (Sawarni et al.) | 0 | 14020 ± 0 | 0% | 0.4340 | 0.3757 |
| LinNash (Sawarni et al.) | -1 | 14020 ± 0 | 0% | 0.4365 | 0.4649 |
| LinNash (Sawarni et al.) | -2 | 14020 ± 0 | 0% | 0.4390 | 0.6663 |
| LinNash (Sawarni et al.) | -5 | 14020 ± 0 | 0% | 0.4467 | 0.8708 |
| LinNash, width constant 1 | 1 | 14020 ± 0 | 100% | 0.4316 | 0.0836 |
| LinNash, width constant 1 | 0 | 14020 ± 0 | 100% | 0.4340 | 0.0935 |
| LinNash, width constant 1 | -1 | 14020 ± 0 | 100% | 0.4364 | 0.1063 |
| LinNash, width constant 1 | -2 | 14020 ± 0 | 100% | 0.4388 | 0.1225 |
| LinNash, width constant 1 | -5 | 14020 ± 0 | 100% | 0.4463 | 0.1943 |
