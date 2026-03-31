# Attack Success Rate by Epsilon

## MNIST

PGD setting: k=40, eps_step=0.01, evaluated on 1000 samples

| Attack | 0.01 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Targeted FGSM | 0.20 | 0.49 | 1.56 | 12.89 | 39.26 | 41.41 |
| Untargeted FGSM | 1.37 | 4.88 | 15.43 | 55.37 | 84.57 | 93.16 |
| Targeted PGD | 0.20 | 0.59 | 3.81 | 69.34 | 99.90 | 100.00 |
| Untargeted PGD | 1.37 | 6.25 | 26.56 | 98.44 | 100.00 | 100.00 |

## CIFAR-10

PGD setting: k=10, eps_step=0.01, evaluated on 1000 samples

| Attack | 0.01 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Targeted FGSM | 23.73 | 12.70 | 8.59 | 8.20 | 8.79 | 8.20 |
| Untargeted FGSM | 78.32 | 87.40 | 89.65 | 90.53 | 89.65 | 90.23 |
| Targeted PGD | 75.29 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Untargeted PGD | 99.12 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
