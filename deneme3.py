import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('101_1b1_Al_sc_AKGC417L.wav')
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectogram)
plt.imshow(spectogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()