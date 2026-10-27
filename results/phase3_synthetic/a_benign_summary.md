| algorithm | p | exploration rounds | p-mean regret at t=1e4 | p-mean regret at T |
|---|---|---|---|---|
| LinUCB | 1 | 0 | 0.0258 | 0.0041 |
| LinUCB | 0 | 0 | 0.0262 | 0.0042 |
| LinUCB | -1 | 0 | 0.0266 | 0.0042 |
| LinUCB | -2 | 0 | 0.0270 | 0.0043 |
| LinUCB | -5 | 0 | 0.0285 | 0.0044 |
| Greedy (ridge) | 1 | 0 | 0.1483 | 0.1483 |
| Greedy (ridge) | 0 | 0 | 0.1483 | 0.1483 |
| Greedy (ridge) | -1 | 0 | 0.1483 | 0.1483 |
| Greedy (ridge) | -2 | 0 | 0.1483 | 0.1483 |
| Greedy (ridge) | -5 | 0 | 0.1483 | 0.1483 |
| Explore-then-LinUCB, naive rule | 1 | 2843 | 0.0580 | 0.0051 |
| Explore-then-LinUCB, naive rule | 0 | 2843 | 0.0612 | 0.0054 |
| Explore-then-LinUCB, naive rule | -1 | 2843 | 0.0647 | 0.0057 |
| Explore-then-LinUCB, naive rule | -2 | 10399 | 0.1778 | 0.0136 |
| Explore-then-LinUCB, naive rule | -5 | 61316 | 0.1786 | 0.0805 |
| Explore-then-LinUCB (ours) | 1 | 10251 | 0.2144 | 0.0119 |
| Explore-then-LinUCB (ours) | 0 | 10251 | 0.2146 | 0.0133 |
| Explore-then-LinUCB (ours) | -1 | 10251 | 0.2148 | 0.0148 |
| Explore-then-LinUCB (ours) | -2 | 38481 | 0.2152 | 0.0558 |
| Explore-then-LinUCB (ours) | -5 | 200000 | 0.2157 | 0.2157 |
| Explore-then-Greedy (ours) | 1 | 10292 | 0.2142 | 0.0117 |
| Explore-then-Greedy (ours) | 0 | 10292 | 0.2144 | 0.0130 |
| Explore-then-Greedy (ours) | -1 | 10292 | 0.2147 | 0.0146 |
| Explore-then-Greedy (ours) | -2 | 38373 | 0.2151 | 0.0559 |
| Explore-then-Greedy (ours) | -5 | 200000 | 0.2159 | 0.2158 |
