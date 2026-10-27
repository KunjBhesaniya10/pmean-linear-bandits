| algorithm | p | exploration rounds | p-mean regret at t=1e4 | p-mean regret at T |
|---|---|---|---|---|
| LinUCB | 1 | 0 | 0.0258 | 0.0041 |
| LinUCB | 0 | 0 | 0.0261 | 0.0042 |
| LinUCB | -1 | 0 | 0.0264 | 0.0042 |
| LinUCB | -2 | 0 | 0.0267 | 0.0042 |
| LinUCB | -5 | 0 | 0.0280 | 0.0044 |
| Greedy (ridge) | 1 | 0 | 0.1599 | 0.1599 |
| Greedy (ridge) | 0 | 0 | 0.1599 | 0.1599 |
| Greedy (ridge) | -1 | 0 | 0.1599 | 0.1599 |
| Greedy (ridge) | -2 | 0 | 0.1599 | 0.1599 |
| Greedy (ridge) | -5 | 0 | 0.1599 | 0.1599 |
| Explore-then-LinUCB, naive rule | 1 | 2840 | 0.0580 | 0.0052 |
| Explore-then-LinUCB, naive rule | 0 | 2840 | 0.0612 | 0.0055 |
| Explore-then-LinUCB, naive rule | -1 | 2840 | 0.0647 | 0.0058 |
| Explore-then-LinUCB, naive rule | -2 | 10405 | 0.1777 | 0.0135 |
| Explore-then-LinUCB, naive rule | -5 | 61268 | 0.1785 | 0.0804 |
| Explore-then-LinUCB (ours) | 1 | 10257 | 0.2144 | 0.0120 |
| Explore-then-LinUCB (ours) | 0 | 10257 | 0.2146 | 0.0133 |
| Explore-then-LinUCB (ours) | -1 | 10257 | 0.2149 | 0.0149 |
| Explore-then-LinUCB (ours) | -2 | 38507 | 0.2152 | 0.0559 |
| Explore-then-LinUCB (ours) | -5 | 200000 | 0.2158 | 0.2158 |
| Explore-then-Greedy (ours) | 1 | 10235 | 0.2144 | 0.0117 |
| Explore-then-Greedy (ours) | 0 | 10235 | 0.2146 | 0.0130 |
| Explore-then-Greedy (ours) | -1 | 10235 | 0.2148 | 0.0146 |
| Explore-then-Greedy (ours) | -2 | 38472 | 0.2151 | 0.0559 |
| Explore-then-Greedy (ours) | -5 | 200000 | 0.2158 | 0.2158 |
