import matplotlib.pyplot as plt
from scipy.io import wavfile 
import scipy   
import numpy as np
import wavio

#rate, data = scipy.io.wavfile.read('101_1b1_Al_sc_AKGC417L.wav')
#filePath="/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_small_copy"


#for filetest in argv[1:]:
#    [fs, x] = wavfile.read(filetest)
#    print ('\nReading with scipy.io.wavfile.read: ', x)
    
def read_waw():
    print('readıng audio')
    rate = 22050  # samples per second
    T = 3         # sample duration (seconds)
    f = 440.0     # sound frequency (Hz)
    t = np.linspace(0, T, T*rate, endpoint=False)
    x = np.sin(2*np.pi * f * t)

    obj = wavio.read("101_1b1_Al_sc_AKGC417L.wav")
    print('read')


if __name__ == "__main__":
    read_waw()
