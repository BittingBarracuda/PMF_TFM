from pmf import ProbabilisticMatrixFactorization
import numpy as np
from scipy.sparse import random

if __name__ == "__main__":
    pmf = ProbabilisticMatrixFactorization(D=10, sigma=0.1, sigma_u=0.1, sigma_v=0.1, max_epochs=1)
    X = np.array(random(10, 10, density=0.3).A)
    print(f'X ->\n{X}')
    pmf.fit(X)
    print(f'R ->\n{pmf.R}')