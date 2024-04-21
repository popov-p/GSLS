import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def ker(x_i: np.ndarray, x_j:np.ndarray, sigma:float=0.8) -> float:
    """
    Computes the Gaussian kernel between two data points x and y.

    Parameters:
    x_i : numpy.ndarray
        Input data point.
    x_j : numpy.ndarray
        Input data point.
    sigma : float, optional
        Kernel parameter controlling the spread of the kernel (default is 0.8).

    Returns:
    float: The Gaussian kernel similarity between x_i and x_j.
    """
    distance = np.sum((x_i - x_j) ** 2)

    kernel_value = np.exp(-distance / (2 * sigma ** 2))

    return kernel_value

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate RMSE (Root Mean Square Error).
    
    Parameters:
        y_true: numpy array, real values
        y_pred: numpy array, predicted values
    
    Returns:
        float: RMSE between y_true and y_pred
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

class GSLS:
    def __init__(self, x_train:list[np.ndarray], y_train:np.ndarray, gamma:float):
        assert len(x_train) == len(y_train), "Length of x_train and y_train must be equal"
        self.D = {i: (x_i, y_i) for i, (x_i, y_i) in enumerate(zip(x_train, y_train), start=1)}
        self.l = len(self.D)
        self.gamma = gamma
        self.S = OrderedDict() # {idx(int): x_{idx}}
        self.beta = OrderedDict() # {idx(int): beta_{idx}}
        self.b = None
        self.RMSES = {}
    def Omega(self)-> np.ndarray:
        S = self.S
        l = self.l
        gamma = self.gamma
        D = self.D

        Omega = np.zeros((len(S.keys()), len(S.keys())))
        for i, key_i in enumerate(S.keys()):
            for j, key_j in enumerate(S.keys()):
                Omega[i][j] = (l/(2*gamma))*ker(S[key_i], S[key_j]) + sum(ker(D[r][0], S[key_j])*ker(D[r][0], S[key_i]) for r in range(1, l+1))
        return Omega

    def Phi(self) -> np.ndarray:
        S = self.S
        l = self.l
        D = self.D
        Phi = np.zeros((len(S.keys()), 1))
        for i, key_i in enumerate(S.keys()):
            Phi[i, 0] = sum(ker(S[key_i], D[j][0]) for j in range(1, l+1))
        #print(Phi)
        return Phi

    def c(self) -> np.ndarray:
        S = self.S
        l = self.l
        D = self.D
        c = np.zeros((len(S.keys()),1))
        for i, key_i in enumerate(S.keys()):
            c[i, 0] = sum(D[j][1]*ker(S[key_i], D[j][0]) for j in range(1, l+1))
        return c
    def L(self) -> float:
        S = self.S
        l = self.l
        beta = self.beta
        gamma = self.gamma
        b = self.b
        D = self.D
        
        L = 0
        left_term = 0
        for i in S.keys():
            for j in S.keys():
                left_term += 0.5*(beta[i]*beta[j]*ker(S[i], S[j]))
        
        right_term = 0 
        for i in range(1, l+1):
            for j in S.keys():
                right_term += (gamma/l)*(D[i][1]-beta[j]*ker(D[i][0], S[j]) - b)**2
        L = left_term + right_term
        return L

    def predict(self, maxvec:int = 6, epsilon: float = 1e-3, criteria:str = 'maxvec'):
            S = self.S
            l = self.l
            D = self.D
            beta = self.beta
            RMSES = self.RMSES
            
            if criteria == 'maxvec':
                for vec in range(maxvec):
                    L_values = {} #{idx: L-value after appending vec-th support vector}
                    all_betas = {}
                    for greedy_iterator in range(1, l+1): # SVM construction in greedy manner
                        if(greedy_iterator not in S.keys()):
                            S[greedy_iterator] = D[greedy_iterator][0]
                            H = np.block([[self.Omega(), self.Phi()],
                                        [np.transpose(self.Phi()), l]])

                            sol = np.linalg.solve(H, np.vstack((self.c(), sum(D[k][1] for k in range(1, l+1)))))
                            self.b = sol[-1]
                            #if not isinstance(sol, list):
                            #    sol = [sol]

                            for idx, beta_idx in zip(S.keys(), reversed(sol[:-1])): #beta_idx - beta by index idx. ex: beta_1 = ... etc..
                                beta[idx] = beta_idx
                                all_betas[idx] = beta_idx
                            L_values[greedy_iterator] = (self.L())

                            del S[greedy_iterator]
                            del beta[greedy_iterator]

                    min_idx = min(L_values, key=L_values.get)
                    S[min_idx] = D[min_idx][0]
                    beta[min_idx] = all_betas[min_idx]

                    RMSES[vec] = rmse(np.array([value[1] for value in D.values()]), np.array([self.regressor(x[0]) for x in D.values()]))
                    print(RMSES[vec])

    def regressor(self, x:np.ndarray):
        beta = self.beta
        S = self.S
        b = self.b
        value = 0 
        for i in beta.keys():
            value += beta[i]*ker(S[i][0], x)

        return value + b
    
    def plot(self):
        D = self.D
        x_train = [pair[0] for pair in D.values()]
        y_train = [pair[1] for pair in D.values()]
        plt.plot(x_train, y_train, marker='o', linestyle='-', label = 'Train')
        plt.plot(x_train, np.array([self.regressor(x[0]) for x in D.values()]), marker='x', linestyle='-', label = 'Predicted')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Without noise')
        plt.grid(True)
        plt.legend()
        plt.show()
        