import numpy as np

class ProbabilisticMatrixFactorization:
    def __init__(self, 
                    D = 30,
                    sigma = 0.3, 
                    sigma_u = 0.3, 
                    sigma_v = 0.3, 
                    learning_rate = 0.01, 
                    max_epochs = 1000
                ):
        self.D = D
        self.sigma = sigma
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    
    def fit(self, R):
        self.R = R
        self.N, self.M = self.R.shape
        self.U, self.V = self.__get_initial_estimation()

    def __get_initial_estimation(self):
        return np.zeros(shape=(self.N, self.D)), np.zeros(shape=(self.D, self.M))
    
    def __gradient_descent(self):
        for t in range(self.max_epochs):
            for i in range(self.N):
                self.U[i, :] = self.U[i, :] - self.learning_rate * self.__get_U_gradient(i)
            for j in range(self.M):
                self.V[:, j] = self.V[:, j] - self.learning_rate * self.__get_V_gradient(j)            
    
    def __get_U_gradient(self, i):
        I_ij = np.copy(self.R[i, :])
        I_ij[np.where(I_ij > 0)] = 1
        aux_1 = I_ij * (self.R[i, :] - np.dot(self.U[i, :], self.V))
        aux_2 = aux_1 * self.V
        return (1 / self.sigma) * np.sum(aux_2, axis = 1) + (self.U[i, :] / self.sigma_u)
