| algorithm | p | κ | exploration rounds | aggregate utility | p-mean regret, p=1 | p-mean regret, p=0 | p-mean regret, p=-2 | worst gender group | worst age group | worst occupation group | group welfare M_-2 over all 30 groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Explore-then-LinUCB, naive rule | 0 | 10 | 34040 | 0.8213 | 0.1578 | 0.1635 | 0.1741 | 0.820 (male) | 0.808 (age 56+) | 0.800 (retired) | 0.8190 |
| Explore-then-LinUCB, naive rule | -2 | 10 | 50000 | 0.7556 | 0.2158 | 0.2168 | 0.2190 | 0.752 (male) | 0.741 (age 56+) | 0.726 (writer) | 0.7586 |
| Explore-then-LinUCB, naive rule | 0 | 1 | 1178 | 0.9491 | 0.0449 | 0.0493 | 0.0651 | 0.938 (female) | 0.924 (age 50-55) | 0.845 (farmer) | 0.9346 |
| Explore-then-LinUCB, naive rule | -2 | 1 | 2688 | 0.9449 | 0.0487 | 0.0530 | 0.0664 | 0.936 (female) | 0.921 (age 56+) | 0.845 (farmer) | 0.9310 |
| Explore-then-LinUCB (ours) | 0 | 10 | 50000 | 0.7285 | 0.2397 | 0.2523 | 0.2861 | 0.728 (female) | 0.702 (age 56+) | 0.680 (writer) | 0.7300 |
| Explore-then-LinUCB (ours) | -2 | 10 | 50000 | 0.7286 | 0.2397 | 0.2520 | 0.2853 | 0.728 (female) | 0.702 (age 56+) | 0.681 (writer) | 0.7301 |
| Explore-then-LinUCB (ours) | 0 | 1 | 5530 | 0.9386 | 0.0543 | 0.0604 | 0.0792 | 0.930 (female) | 0.908 (age 56+) | 0.741 (farmer) | 0.9137 |
| Explore-then-LinUCB (ours) | -2 | 1 | 13217 | 0.9082 | 0.0811 | 0.0918 | 0.1242 | 0.899 (female) | 0.862 (age 56+) | 0.722 (unemployed) | 0.8701 |
| Explore-then-LinUCB, one stop for all groups | 0 | 10 | 50000 | 0.7284 | 0.2398 | 0.2524 | 0.2868 | 0.728 (female) | 0.703 (age 56+) | 0.680 (writer) | 0.7300 |
| Explore-then-LinUCB, one stop for all groups | -2 | 10 | 50000 | 0.7286 | 0.2397 | 0.2521 | 0.2856 | 0.728 (female) | 0.703 (age 56+) | 0.681 (writer) | 0.7303 |
| Explore-then-LinUCB, one stop for all groups | 0 | 1 | 50000 | 0.7285 | 0.2397 | 0.2523 | 0.2869 | 0.729 (male) | 0.703 (age 56+) | 0.681 (writer) | 0.7304 |
| Explore-then-LinUCB, one stop for all groups | -2 | 1 | 50000 | 0.7284 | 0.2399 | 0.2524 | 0.2868 | 0.728 (female) | 0.702 (age 56+) | 0.681 (writer) | 0.7301 |
