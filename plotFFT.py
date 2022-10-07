'''compute fft of a signal and plot it'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

x = np.linspace(0, 1, 1000, False)
y = np.sin(50 * 2 * np.pi * x) + 0.5 * np.sin(80 * 2 * np.pi * x)

yf = fft(y)
xf = fftfreq(len(y), 1 / 1000)

# plot in linear scale
plt.plot(xf, np.abs(yf))
# plot in log scale
# plt.plot(xf, 20 * np.log10(np.abs(yf)))
# label the axes
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
# set the title
plt.title('FFT of sine waves')
# set the axis limits
plt.xlim(-100, 100)
plt.ylim(-10, 550)
# display the plot
plt.grid()
plt.show()

'''plot a sin wave'''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 1000, False)
y = np.sin(50 * 2 * np.pi * x) + 0.5 * np.sin(80 * 2 * np.pi * x)

plt.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sin wave')
plt.grid()
plt.show()

