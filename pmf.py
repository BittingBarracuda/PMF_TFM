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
        self.__gradient_descent()
        self.R = np.dot(self.U, self.V)

    def __get_initial_estimation(self):
        return np.zeros(shape=(self.N, self.D)), np.zeros(shape=(self.D, self.M))
    
    def __gradient_descent(self):
        aux_U, aux_V = np.zeros(shape=(self.N, self.D)), np.zeros(shape=(self.D, self.M))
        for t in range(self.max_epochs):
            for i in range(self.N):
                aux_U[i, :] = self.U[i, :] - self.learning_rate * self.__get_U_gradient(i)
            for j in range(self.M):
                aux_V[:, j] = self.V[:, j] - self.learning_rate * self.__get_V_gradient(j)  
            print(f'Epoch-{t}')
            self.U, self.V = aux_U, aux_V
            print(self.U)
            print(self.V)          
    
    def __get_U_gradient(self, i):
        # I_ij -> 1 x M
        I_ij = np.copy(self.R[i, :])
        I_ij[np.where(I_ij > 0)] = 1
        # self.R[i, :] -> 1 x M
        # self.U[i, :] -> 1 x D
        # self.V -> D x M
        # np.dot(self.U[i, :], self.V) -> 1 x M
        # aux_1 -> 1 x M
        # aux_2 -> D x M 
        # sum(aux_2, axis = 1) -> D x 1
        aux_1 = I_ij * (self.R[i, :] - np.dot(self.U[i, :], self.V))
        aux_2 = aux_1 * self.V
        return (1 / self.sigma) * np.sum(aux_2, axis=1) + (self.U[i, :] / self.sigma_u)

    def __get_V_gradient(self, j):
        # I_ij -> 1 x M
        I_ij = np.copy(self.R[:, j])
        I_ij[np.where(I_ij > 0)] = 1
        # self.R[:, j] -> 1 x N
        # self.U -> N x D
        # self.V[:, j] -> 1 x D
        # np.dot(self.U, self.V[:, j]) -> N x 1
        # aux_1 -> N x 1
        # aux_2 -> N x D 
        # sum(aux_2, axis = 1) -> 1 x D
        aux_1 = I_ij * (self.R[:, j] - np.dot(self.U, self.V[:, j]))
        aux_2 = aux_1 * self.U.T
        return (1 / self.sigma) * np.sum(aux_2, axis=1) + (self.V[:, j] / self.sigma_v)