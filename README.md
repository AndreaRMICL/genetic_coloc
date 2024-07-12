# genetic_coloc 🧬

An implementation of the Approximate Bayes Factor (ABF) fine mapping and colocalisation methods [1].

## Installation

To install the `genetic_coloc` package, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/AndreRMICL/genetic_coloc.git
    ```

2. Navigate to the repository directory:
    ```sh
    cd genetic_coloc
    ```

3. Install the package using `pip`:
    ```sh
    pip install .
    ```

## Quickstart

### Compute log ABF

```python
import numpy as np

from genetic_coloc import compute_log_abf

beta = np.array([0.1, 0.2, 0.3])
se = np.array([0.01, 0.02, 0.03])
compute_log_abf(beta=beta, se=se, se_prior=0.20)
```

### Fine mapping

```python
import numpy as np

from genetic_coloc import finemapping_abf

log_abf = np.array([46.87, 47.19, 46.99])
prior = np.array([1e-04, 1e-04, 1e-04])
finemapping_abf(log_abf=log_abf, prior=prior)
```

### Colocalisation

```python
import numpy as np

from genetic_coloc import coloc_abf

log_abf_trait_1 = np.array([1, 5, 10])
log_abf_trait_2 = np.array([1, 4, 9])
coloc_abf(log_abf_trait_1=log_abf_trait_1, log_abf_trait_2=log_abf_trait_2)
```

## References
[1] https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004383.

