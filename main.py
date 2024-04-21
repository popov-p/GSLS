from GSLS import GSLS, ker
import numpy as np

x_train = np.linspace(0,5, num=200)
y_train = np.sinc(x_train)

def main():
    model = GSLS([np.array([x]) for x in x_train], y_train, gamma=float(3e4))
    model.predict(maxvec=4, criteria='maxvec')
    model.plot()
    
    return 0
if __name__ == '__main__':
    main()