| algorithm | p | exploration rounds (mean ± sd) | runs ending on the optimal arm | p-mean regret at t=1e4 | p-mean regret at T |
|---|---|---|---|---|---|
| Explore-then-LinUCB, naive rule | 1 | 5197 ± 139 | 100% | 0.2029 | 0.0104 |
| Explore-then-LinUCB, naive rule | 0 | 5197 ± 139 | 100% | 0.2271 | 0.0132 |
| Explore-then-LinUCB, naive rule | -1 | 5197 ± 139 | 100% | 0.2503 | 0.0170 |
| Explore-then-LinUCB, naive rule | -2 | 19119 ± 264 | 100% | 0.3880 | 0.0742 |
| Explore-then-LinUCB, naive rule | -5 | 112792 ± 729 | 100% | 0.3889 | 0.3269 |
| design only (naive rule) | 1 | 6488 ± 159 | 100% | 0.2622 | 0.0133 |
| design only (naive rule) | 0 | 6488 ± 159 | 100% | 0.2869 | 0.0171 |
| design only (naive rule) | -1 | 6488 ± 159 | 100% | 0.3089 | 0.0225 |
| design only (naive rule) | -2 | 23508 ± 304 | 100% | 0.4078 | 0.0976 |
| design only (naive rule) | -5 | 138419 ± 740 | 100% | 0.4121 | 0.3733 |
| coverage rule only (uniform) | 1 | 200000 ± 0 | 0% | 0.3871 | 0.3871 |
| coverage rule only (uniform) | 0 | 200000 ± 0 | 0% | 0.3874 | 0.3873 |
| coverage rule only (uniform) | -1 | 200000 ± 0 | 0% | 0.3877 | 0.3876 |
| coverage rule only (uniform) | -2 | 200000 ± 0 | 0% | 0.3878 | 0.3879 |
| coverage rule only (uniform) | -5 | 200000 ± 0 | 0% | 0.3888 | 0.3889 |
| Explore-then-LinUCB (ours) | 1 | 15582 ± 370 | 100% | 0.4039 | 0.0315 |
| Explore-then-LinUCB (ours) | 0 | 15582 ± 370 | 100% | 0.4052 | 0.0401 |
| Explore-then-LinUCB (ours) | -1 | 15582 ± 370 | 100% | 0.4065 | 0.0519 |
| Explore-then-LinUCB (ours) | -2 | 56747 ± 840 | 100% | 0.4080 | 0.1951 |
| Explore-then-LinUCB (ours) | -5 | 200000 ± 0 | 0% | 0.4120 | 0.4119 |
| Explore-then-Greedy, naive rule | 1 | 5274 ± 135 | 52% | 0.2277 | 0.0580 |
| Explore-then-Greedy, naive rule | 0 | 5274 ± 135 | 52% | 0.2470 | 0.0601 |
| Explore-then-Greedy, naive rule | -1 | 5274 ± 135 | 52% | 0.2655 | 0.0631 |
| Explore-then-Greedy, naive rule | -2 | 19116 ± 255 | 78% | 0.3879 | 0.0899 |
| Explore-then-Greedy, naive rule | -5 | 112724 ± 613 | 100% | 0.3891 | 0.3269 |
| Explore-then-Greedy (ours) | 1 | 15521 ± 403 | 100% | 0.4039 | 0.0313 |
| Explore-then-Greedy (ours) | 0 | 15521 ± 403 | 100% | 0.4052 | 0.0400 |
| Explore-then-Greedy (ours) | -1 | 15521 ± 403 | 100% | 0.4066 | 0.0517 |
| Explore-then-Greedy (ours) | -2 | 56625 ± 669 | 100% | 0.4079 | 0.1950 |
| Explore-then-Greedy (ours) | -5 | 200000 ± 0 | 0% | 0.4120 | 0.4120 |
| LinUCB | 1 | 0 ± 0 | 100% | 0.0170 | 0.0012 |
| LinUCB | 0 | 0 ± 0 | 100% | 0.0177 | 0.0012 |
| LinUCB | -1 | 0 ± 0 | 100% | 0.0185 | 0.0013 |
| LinUCB | -2 | 0 ± 0 | 100% | 0.0197 | 0.0013 |
| LinUCB | -5 | 0 ± 0 | 100% | 0.0314 | 0.0021 |
| Greedy (ridge) | 1 | 0 ± 0 | 2% | 0.2739 | 0.2735 |
| Greedy (ridge) | 0 | 0 ± 0 | 2% | 0.2739 | 0.2735 |
| Greedy (ridge) | -1 | 0 ± 0 | 2% | 0.2739 | 0.2735 |
| Greedy (ridge) | -2 | 0 ± 0 | 2% | 0.2739 | 0.2735 |
| Greedy (ridge) | -5 | 0 ± 0 | 2% | 0.2740 | 0.2735 |
