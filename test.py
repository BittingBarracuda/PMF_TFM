from pmf import ProbabilisticMatrixFactorization
import numpy as np
from scipy.sparse import random

if __name__ == "__main__":
    pmf = ProbabilisticMatrixFactorization(D=10, sigma=0.1, sigma_u=0.1, sigma_v=0.1, max_epochs=100)
    X = np.array(random(1000, 10, density=0.3).A)
    pmf.fit(X)
    print(pmf.R)