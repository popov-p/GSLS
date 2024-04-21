import numpy as np

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
    def __init__(x_train:list[np.ndarray], y_train:list, gamma:float):
        assert len(x_train) == len(y_train), "Length of x_train and y_train must be equal"
        self.D = {i: (x_i, y_i) for i, (x_i, y_i) in enumerate(zip(x_train, y_train), start=1)}
        self.l = len(D)
        self.gamma = gamma
        self.S = {} # {idx(int): x_{idx}}
        self.beta = {} # {idx(int): beta_{idx}}
    def Omega():
        Omega = np.empty(max(self.S.keys()), max(self.S.keys()))
        for i in sorted(self.S.keys()):
            for j in sorted(self.S.keys()):
                Omega[i][j] = (self.l/(2*self.gamma))*ker(D[i][0]-D[j][0]) + sum(ker(D[r][0], D[j][0])*ker(D[r][0], D[i][0]) for r in range(1, self.l+1))
        return Omega

    def Phi():
        Phi = np.empty(max(self.S.keys()), max(self.S.keys()))
        for i in sorted(self.S.keys()):
            for j in range(1, self.l+1):
                Phi[i] = sum(ker(D[i][0], D[j][0]))
        return Phi

    def c():
        for i in sorted(self.S.keys()):
            for j in range(1, self.l+1):
                c[j] = D[j][1]*ker(D[i][0], D[j][0])
        return c
    def L(beta:dict, b:float) -> float:
        """
        Calculate GSLS-SVM objective function
        
        Parameters:
            beta:list, predicted regression coefficietns
            S:list, (set?) of support vector indices

        Returns:
            float: GSLS-SVM objective function value
        """
        L = None
        left_term = None
        for i in S:
            for j in S:
                left_term += 0.5*(beta(i)*beta(j)*ker(D[i][0], D[j][0]))
        
        right_term = None 
        for i in range(1, self.l+1):
            for j in S:
                right_term += (self.gamma/self.l)*(D[i][1]-beta(j)*ker(D[i][0], D[j][0]) - b)**2
        L = left_term + right_term
        return L
    def regressor(beta:dict, ):
        for in
        w = beta[i]
        return 
    def predict(maxvec:int = 7, epsilon: float = 1e-3, criteria:string = 'maxvec'):
            RMSES = {'i':[0.2, 0.3, 0.4]}
            if criteria == 'maxvec':
                for i in range(maxvec):
                    for greedy_iterator in range(1, self.l+1): # SVM construction in greedy manner
                        self.S.append(self.D[greedy_iterator][0])
                        H = np.block([[Omega, Phi],
                                    [np.transpose(Phi), self.l]])
                        beta, b = np.linalg.solve(H, np.block(c, sum(D[k][1] for k in range(1, self.l+1))))

                        #rmse(np.array[value[1] for value in self.D.values()], L[beta, b])