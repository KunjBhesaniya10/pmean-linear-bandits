| algorithm | p | exploration rounds (mean ± sd) | runs ending on the optimal arm | p-mean regret at t=1e4 | p-mean regret at T |
|---|---|---|---|---|---|
| Explore-then-LinUCB, naive rule | 1 | 5242 ± 149 | 100% | 0.2045 | 0.0105 |
| Explore-then-LinUCB, naive rule | 0 | 5242 ± 149 | 100% | 0.2285 | 0.0132 |
| Explore-then-LinUCB, naive rule | -1 | 5242 ± 149 | 100% | 0.2515 | 0.0171 |
| Explore-then-LinUCB, naive rule | -2 | 19073 ± 235 | 100% | 0.3878 | 0.0740 |
| Explore-then-LinUCB, naive rule | -5 | 112833 ± 711 | 100% | 0.3890 | 0.3270 |
| design only (naive rule) | 1 | 6466 ± 159 | 100% | 0.2613 | 0.0132 |
| design only (naive rule) | 0 | 6466 ± 159 | 100% | 0.2862 | 0.0170 |
| design only (naive rule) | -1 | 6466 ± 159 | 100% | 0.3083 | 0.0225 |
| design only (naive rule) | -2 | 23442 ± 325 | 100% | 0.4079 | 0.0973 |
| design only (naive rule) | -5 | 138740 ± 700 | 100% | 0.4121 | 0.3736 |
| coverage rule only (uniform) | 1 | 200000 ± 0 | 0% | 0.3871 | 0.3870 |
| coverage rule only (uniform) | 0 | 200000 ± 0 | 0% | 0.3874 | 0.3873 |
| coverage rule only (uniform) | -1 | 200000 ± 0 | 0% | 0.3877 | 0.3876 |
| coverage rule only (uniform) | -2 | 200000 ± 0 | 0% | 0.3880 | 0.3879 |
| coverage rule only (uniform) | -5 | 200000 ± 0 | 0% | 0.3889 | 0.3888 |
| Explore-then-LinUCB (ours) | 1 | 15587 ± 410 | 100% | 0.4040 | 0.0315 |
| Explore-then-LinUCB (ours) | 0 | 15587 ± 410 | 100% | 0.4053 | 0.0402 |
| Explore-then-LinUCB (ours) | -1 | 15587 ± 410 | 100% | 0.4066 | 0.0520 |
| Explore-then-LinUCB (ours) | -2 | 56702 ± 629 | 100% | 0.4078 | 0.1952 |
| Explore-then-LinUCB (ours) | -5 | 200000 ± 0 | 0% | 0.4116 | 0.4120 |
| Explore-then-Greedy, naive rule | 1 | 5239 ± 141 | 52% | 0.2263 | 0.0573 |
| Explore-then-Greedy, naive rule | 0 | 5239 ± 141 | 52% | 0.2456 | 0.0595 |
| Explore-then-Greedy, naive rule | -1 | 5239 ± 141 | 52% | 0.2643 | 0.0624 |
| Explore-then-Greedy, naive rule | -2 | 19136 ± 263 | 80% | 0.3880 | 0.0885 |
| Explore-then-Greedy, naive rule | -5 | 112780 ± 580 | 100% | 0.3890 | 0.3270 |
| Explore-then-Greedy (ours) | 1 | 15657 ± 373 | 100% | 0.4039 | 0.0316 |
| Explore-then-Greedy (ours) | 0 | 15657 ± 373 | 100% | 0.4052 | 0.0403 |
| Explore-then-Greedy (ours) | -1 | 15657 ± 373 | 100% | 0.4066 | 0.0522 |
| Explore-then-Greedy (ours) | -2 | 56663 ± 706 | 100% | 0.4079 | 0.1950 |
| Explore-then-Greedy (ours) | -5 | 200000 ± 0 | 0% | 0.4121 | 0.4120 |
| LinUCB | 1 | 0 ± 0 | 100% | 0.0170 | 0.0012 |
| LinUCB | 0 | 0 ± 0 | 100% | 0.0177 | 0.0012 |
| LinUCB | -1 | 0 ± 0 | 100% | 0.0186 | 0.0013 |
| LinUCB | -2 | 0 ± 0 | 100% | 0.0197 | 0.0013 |
| LinUCB | -5 | 0 ± 0 | 100% | 0.0308 | 0.0020 |
| Greedy (ridge) | 1 | 0 ± 0 | 2% | 0.2802 | 0.2802 |
| Greedy (ridge) | 0 | 0 ± 0 | 2% | 0.2802 | 0.2802 |
| Greedy (ridge) | -1 | 0 ± 0 | 2% | 0.2803 | 0.2802 |
| Greedy (ridge) | -2 | 0 ± 0 | 2% | 0.2803 | 0.2802 |
| Greedy (ridge) | -5 | 0 ± 0 | 2% | 0.2803 | 0.2802 |
