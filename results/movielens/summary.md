| algorithm | p | κ | exploration rounds | aggregate utility | p-mean regret, p=1 | p-mean regret, p=0 | p-mean regret, p=-2 | worst gender group | worst age group | worst occupation group | group welfare M_-2 over all 30 groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Explore-then-LinUCB, naive rule | 0 | 10 | 50000 | 0.7560 | 0.2154 | 0.2164 | 0.2185 | 0.752 (male) | 0.741 (age 56+) | 0.726 (writer) | 0.7592 |
| Explore-then-LinUCB, naive rule | -2 | 10 | 50000 | 0.7555 | 0.2158 | 0.2169 | 0.2190 | 0.752 (male) | 0.740 (age 56+) | 0.725 (writer) | 0.7586 |
| Explore-then-LinUCB, naive rule | 0 | 1 | 2950 | 0.9228 | 0.0682 | 0.0767 | 0.1222 | 0.905 (female) | 0.889 (age 56+) | 0.836 (farmer) | 0.9073 |
| Explore-then-LinUCB, naive rule | -2 | 1 | 6784 | 0.9126 | 0.0771 | 0.0847 | 0.1128 | 0.900 (female) | 0.881 (age 56+) | 0.837 (farmer) | 0.8995 |
| Explore-then-LinUCB (ours) | 0 | 10 | 50000 | 0.7244 | 0.2433 | 0.2564 | 0.2909 | 0.724 (female) | 0.696 (age 56+) | 0.676 (writer) | 0.7250 |
| Explore-then-LinUCB (ours) | -2 | 10 | 50000 | 0.7244 | 0.2433 | 0.2563 | 0.2903 | 0.724 (female) | 0.696 (age 56+) | 0.676 (writer) | 0.7251 |
| Explore-then-LinUCB (ours) | 0 | 1 | 12395 | 0.8965 | 0.0914 | 0.1027 | 0.1358 | 0.884 (female) | 0.844 (age 56+) | 0.708 (unemployed) | 0.8607 |
| Explore-then-LinUCB (ours) | -2 | 1 | 27884 | 0.8317 | 0.1486 | 0.1653 | 0.2094 | 0.821 (female) | 0.736 (age 56+) | 0.691 (retired) | 0.7894 |
| Explore-then-LinUCB, one stop for all groups | 0 | 10 | 50000 | 0.7242 | 0.2435 | 0.2567 | 0.2918 | 0.724 (female) | 0.697 (age 56+) | 0.675 (writer) | 0.7250 |
| Explore-then-LinUCB, one stop for all groups | -2 | 10 | 50000 | 0.7244 | 0.2433 | 0.2564 | 0.2906 | 0.724 (female) | 0.697 (age 56+) | 0.675 (writer) | 0.7252 |
| Explore-then-LinUCB, one stop for all groups | 0 | 1 | 50000 | 0.7244 | 0.2434 | 0.2565 | 0.2917 | 0.724 (male) | 0.697 (age 56+) | 0.676 (writer) | 0.7253 |
| Explore-then-LinUCB, one stop for all groups | -2 | 1 | 50000 | 0.7243 | 0.2434 | 0.2566 | 0.2918 | 0.724 (male) | 0.696 (age 56+) | 0.676 (writer) | 0.7251 |
| Explore-then-Greedy (ours) | 0 | 10 | 50000 | 0.7244 | 0.2433 | 0.2564 | 0.2912 | 0.724 (female) | 0.698 (age 56+) | 0.676 (writer) | 0.7252 |
| Explore-then-Greedy (ours) | -2 | 10 | 50000 | 0.7244 | 0.2433 | 0.2564 | 0.2909 | 0.724 (female) | 0.698 (age 56+) | 0.676 (writer) | 0.7251 |
| Explore-then-Greedy (ours) | 0 | 1 | 12578 | 0.9279 | 0.0637 | 0.0754 | 0.1121 | 0.921 (female) | 0.882 (age 56+) | 0.710 (unemployed) | 0.8874 |
| Explore-then-Greedy (ours) | -2 | 1 | 27775 | 0.8440 | 0.1377 | 0.1556 | 0.2029 | 0.834 (female) | 0.742 (age 56+) | 0.691 (retired) | 0.7965 |
