#!/Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os,codecs
from time import sleep
from math import sqrt,log
from scipy import signal,fft
import numpy, matplotlib
from lame  import *
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

mp = re.compile('\.mp3')

files = []
show_plot = False
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
    if (len(sys.argv) > 2): show_plot = True
else:
    files = os.listdir('.')

debug = False
count = 0
sr = 44100.0

for fil in files:
    if (mp.search(fil)):
        
        audio_in = decode_mp3(fil)
        samples = len(audio_in)
        fft_size = 2**int(floor(log(samples)/log(2.0)))
        print 'samples,fft_size',samples,fft_size
        freq = fft(audio_in[0:fft_size])
        s_data = numpy.zeros(fft_size)
        x_data = numpy.zeros(fft_size)
        min_x = log(1.0/fft_size);
        for j in xrange(fft_size):
            x_data[j] = log(1.0*(j+1)/fft_size);
            if (x_data[j] < -10):
                x_data[j] = -10
            s_data[j] = 10.0*log(abs(freq[j]))/log(10.0)
        plt.plot(x_data,s_data)
        plt.title('fft log power')
        plt.grid()
        fields = fil.split('.')
        plt.savefig(fields[0]+'_fft.png', bbox_inches="tight")
        plt.draw()
        plt.waitforbuttonpress(timeout=22.1)
        plt.hold(True)
        


