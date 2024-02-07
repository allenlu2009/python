'''plot a sine wave using matplotlib'''

import matplotlib.pyplot as plt
import numpy as np

def plot_sin():
    '''
    plot a sine wave
    '''
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    plot_sin()