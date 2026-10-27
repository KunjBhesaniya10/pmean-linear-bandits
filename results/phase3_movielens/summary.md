| algorithm | p | κ | exploration rounds | aggregate utility | p-mean regret, p=1 | p-mean regret, p=0 | p-mean regret, p=-2 | worst gender group | worst age group | worst occupation group | group welfare M_-2 over all 30 groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LinUCB | - | - | 0 | 0.9296 | 0.0621 | 0.0724 | 0.1522 | 0.912 (female) | 0.895 (age 56+) | 0.846 (farmer) | 0.9135 |
| Greedy (ridge) | - | - | 0 | 0.9963 | 0.0032 | 0.0033 | 0.0035 | 0.996 (female) | 0.993 (age <18) | 0.986 (homemaker) | 0.9956 |
| Explore-then-LinUCB, naive rule | 0 | 10 | 50000 | 0.7557 | 0.2157 | 0.2167 | 0.2188 | 0.752 (male) | 0.741 (age 56+) | 0.726 (writer) | 0.7589 |
| Explore-then-LinUCB, naive rule | -2 | 10 | 50000 | 0.7557 | 0.2157 | 0.2167 | 0.2188 | 0.752 (male) | 0.740 (age 56+) | 0.725 (writer) | 0.7588 |
| Explore-then-LinUCB, naive rule | 0 | 1 | 2943 | 0.9233 | 0.0677 | 0.0761 | 0.1197 | 0.906 (female) | 0.890 (age 56+) | 0.843 (farmer) | 0.9081 |
| Explore-then-LinUCB, naive rule | -2 | 1 | 6607 | 0.9131 | 0.0767 | 0.0843 | 0.1128 | 0.899 (female) | 0.879 (age 56+) | 0.836 (farmer) | 0.8997 |
| Explore-then-LinUCB (ours) | 0 | 10 | 50000 | 0.7244 | 0.2433 | 0.2564 | 0.2911 | 0.724 (female) | 0.697 (age 56+) | 0.676 (writer) | 0.7252 |
| Explore-then-LinUCB (ours) | -2 | 10 | 50000 | 0.7243 | 0.2434 | 0.2565 | 0.2914 | 0.724 (male) | 0.697 (age 56+) | 0.676 (writer) | 0.7252 |
| Explore-then-LinUCB (ours) | 0 | 1 | 12388 | 0.8969 | 0.0910 | 0.1023 | 0.1351 | 0.884 (female) | 0.847 (age 56+) | 0.710 (unemployed) | 0.8613 |
| Explore-then-LinUCB (ours) | -2 | 1 | 27782 | 0.8320 | 0.1484 | 0.1649 | 0.2089 | 0.821 (female) | 0.735 (age 56+) | 0.690 (retired) | 0.7897 |
| Explore-then-LinUCB, one stop for all groups | 0 | 10 | 50000 | 0.7245 | 0.2432 | 0.2562 | 0.2907 | 0.724 (female) | 0.697 (age 56+) | 0.677 (writer) | 0.7253 |
| Explore-then-LinUCB, one stop for all groups | -2 | 10 | 50000 | 0.7243 | 0.2434 | 0.2565 | 0.2911 | 0.724 (female) | 0.697 (age 56+) | 0.675 (writer) | 0.7251 |
| Explore-then-LinUCB, one stop for all groups | 0 | 1 | 50000 | 0.7244 | 0.2433 | 0.2563 | 0.2907 | 0.724 (female) | 0.696 (age 56+) | 0.677 (writer) | 0.7252 |
| Explore-then-LinUCB, one stop for all groups | -2 | 1 | 50000 | 0.7244 | 0.2433 | 0.2564 | 0.2909 | 0.724 (female) | 0.696 (age 56+) | 0.674 (writer) | 0.7251 |
| Explore-then-Greedy (ours) | 0 | 10 | 50000 | 0.7246 | 0.2432 | 0.2562 | 0.2905 | 0.725 (male) | 0.696 (age 56+) | 0.676 (writer) | 0.7253 |
| Explore-then-Greedy (ours) | -2 | 10 | 50000 | 0.7241 | 0.2436 | 0.2568 | 0.2920 | 0.724 (female) | 0.697 (age 56+) | 0.675 (writer) | 0.7250 |
| Explore-then-Greedy (ours) | 0 | 1 | 12599 | 0.9276 | 0.0639 | 0.0756 | 0.1121 | 0.921 (female) | 0.880 (age 56+) | 0.709 (unemployed) | 0.8871 |
| Explore-then-Greedy (ours) | -2 | 1 | 27740 | 0.8443 | 0.1375 | 0.1553 | 0.2026 | 0.835 (female) | 0.740 (age 56+) | 0.690 (retired) | 0.7972 |
