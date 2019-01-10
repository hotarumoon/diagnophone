import time
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt

x = np.random.randn(10)
print('ready to plot')
plt.plot(x)
plt.pause(0.01)
print('starting to sleep (or working hard)')
time.sleep(2)
plt.plot(x + 2)
plt.pause(0.01)
print('sleeping again (or more work)')
time.sleep(2)
print('now blocking until the figure is closed')
plt.show(block=True)
