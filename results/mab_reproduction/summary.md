| expt | algorithm | p | T | regret at t=1e4 | regret at T | exploration rounds (mean ± sd) |
|---|---|---|---|---|---|---|
| a | ncb | 0 | 1e+07 | 0.5239 | 0.225 | 3349088 ± 11169 |
| a | welfarist_ucb | 0 | 1e+07 | 0.5232 | 0.06186 | 828816 ± 2328 |
| b | ncb | 0 | 1e+06 | 156.2 | 1.75 | 2487 ± 202 |
| b | welfarist_ucb | 0 | 1e+06 | 17.98 | 0.182 | 266 ± 12 |
| c | explore_then_ucb | 0.5 | 1e+06 | 489.4 | 45.33 | 79955 ± 0 |
| c | welfarist_ucb | 0.5 | 1e+06 | 15.76 | 0.16 | 266 ± 11 |
| d | explore_then_ucb | -0.5 | 1e+06 | 491.1 | 18.08 | 22365 ± 0 |
| d | welfarist_ucb | -0.5 | 1e+06 | 21.79 | 0.2233 | 266 ± 12 |
| e | explore_then_ucb | -1.5 | 1e+06 | 260.8 | 3.854 | 3163 ± 0 |
| e | welfarist_ucb | -1.5 | 1e+06 | 65.74 | 0.7188 | 592 ± 21 |
| f | welfarist_ucb | 1 | 1e+06 | 12.9 | 0.1292 | 276 ± 14 |
| f | welfarist_ucb | 0 | 1e+06 | 16.95 | 0.1712 | 273 ± 14 |
| f | welfarist_ucb | -1 | 1e+06 | 23.47 | 0.2406 | 275 ± 14 |
| f | welfarist_ucb | -2 | 1e+06 | 108.3 | 1.29 | 1010 ± 26 |
| f | welfarist_ucb | -5 | 1e+06 | 418.9 | 27.01 | 5922 ± 30 |

Instances (mu*):

- (a) k=50, mu*=0.9701, min mean=0.02548, sigma^2=1.0
- (b) k=50, mu*=987, min mean=15.47, sigma^2=400.0
- (c) k=50, mu*=972.1, min mean=16.88, sigma^2=400.0
- (d) k=50, mu*=972.1, min mean=16.88, sigma^2=400.0
- (e) k=50, mu*=972.1, min mean=16.88, sigma^2=400.0
- (f) k=50, mu*=985.8, min mean=26.42, sigma^2=400.0
