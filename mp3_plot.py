#!/Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re
import sys,os
import codecs
from lame import decode_mp3
from time import sleep
from math import sqrt,log
from scipy.io.wavfile import read,write
from scipy import signal,fft
import numpy
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt

# Just plot waveform (assuming mono)
mp = re.compile('\.mp3')

files = []
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
else:
    files = os.listdir('.')

count = 0
for fil in files:
    if (mp.search(fil)):
        audio_in = decode_mp3(fil)
        plt.figure()
        plt.plot(audio_in,'r.')
        plt.pause(1.1)

plt.pause(5)




