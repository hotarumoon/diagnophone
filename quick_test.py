#!/Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os,codecs
from time import sleep
from math import sqrt,log,pi,sin,atan2
import cmath
from scipy import signal,fft
import numpy, matplotlib
from lame  import *
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from tone_est import *

show_plot = False
debug = False
count = 0
points = 44100/50

eps = 1e-3 # -60 dB for plots

sr = 44100
samples = 44100
audio_in = numpy.zeros(samples)

show_t = False
save_files = False
f0 = 50/44100.0
phase1 = 1; # radians
phase2 = 2; # radians
phase3 = 3; # radians
phase4 = 4; # radians
phase5 = 5; # radians

for i in xrange(samples):
    audio_in[i] = sin(2*pi*f0*i+phase1) - 0.000005*sin(2*pi*2*f0*i+phase2)
    #+ 0.25*sin(2*pi*3*f0*i+phase3) - 0.125*sin(2*pi*4*f0*i+phase4) + 0.0625*sin(2*pi*5*f0*i+phase5);


(a,f,p) = tone_est(audio_in,sr)
print "(a,f,p) = ",a,f,p
f = 50.0
#a = 1.0
adata = est_tone_phase_and_remove(audio_in,a,f,sr)


#plt.xlim([0,4000]);
plt.plot(audio_in, 'r')
plt.hold(True)
plt.grid(True)
plt.plot(adata, 'b')
plt.show()

