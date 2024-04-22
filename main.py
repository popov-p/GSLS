from GSLS import GSLS, ker
import numpy as np

x_train = np.linspace(-3,3, num=200)
y_train = np.sin(x_train)

noise_level = 0.3
random_noise = np.random.normal(loc=0, scale=noise_level, size=y_train.shape)
y_train_noisy = y_train + random_noise

def main():
    model = GSLS([np.array([x]) for x in x_train],
                 y_train, 
                 gamma=float(100000)
                )
    model.predict(maxvec=5, criteria='maxvec')
    model.plot()
    model.plot_rmse()
    return 0
if __name__ == '__main__':
    main()