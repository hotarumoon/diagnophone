#!/Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os,codecs
from time import sleep
from math import sqrt,log,pi,sin,cos,atan2
import cmath 
from scipy import signal,fft
import numpy, matplotlib
from lame  import *
from tone_est import *
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt

mp = re.compile('\.mp3')

sr = 44100.0
sr2 = sr/2.0
samples = 16834
gain = 0.1
f = [121.12]
offset = 0*pi/8.0
show_plot = False

sdata  = numpy.zeros(samples)
for i in xrange(samples): sdata[i] = gain*cos(2*pi*i*f[0]/sr2 + offset)

(amp,freq_est,phase_r) = tone_est(sdata,sr)

ddata = est_tone_and_remove(sdata,sr)

print "amp = ",amp, "freq = ",freq_est, "phase (degrees) = ",180*phase_r/pi

plt.plot(sdata[0:200], 'r')
plt.hold(True)
plt.grid(True)
plt.plot(ddata[0:200], 'g')
plt.waitforbuttonpress(timeout=100.1)

