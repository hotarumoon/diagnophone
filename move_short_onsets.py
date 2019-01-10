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

show = False
files = []
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
    show = True
else:
    files = os.listdir('.')

b_lpf, a_lpf = signal.butter(1, 200/44100.0, 'low')

check = 2000
thres = 0.015
count = 0
for fil in files:
    if (mp.search(fil)):
        audio_in = decode_mp3(fil)
        sim = numpy.square(audio_in)
        if (len(sim) < 100):
            print "Wow, too short ",fil, ' len = ',len(sim)
        else:
            audio_lpf   = signal.filtfilt(b_lpf,a_lpf,sim)
            found = 0
            short = 0
            for i in xrange(len(audio_in)):
                if ((audio_lpf[i] > thres) and not found):
                    if (i < 4000):
                        print "Detected onset for ",fil," at time ",i/44.1," msecs "
                        cmd = 'mv \"'+fil+'\" ./short/\"'+fil+'\"'
                        os.system(cmd)
                        short = 1
                    found = 1
            if (not short):
                cmd = 'rm \"'+fil+'\"'
                os.system(cmd)
                
                





