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

mp = re.compile('\.mp3')

fil = sys.argv[1]

debug = False
count = 0
sr = 44100.0

f = [100];
points = 44100/f[0]

eps = 1e-3 # -60 dB for plots
#offset = pi*(120.)/180 - orig
offset = pi*(195.)/180
gain = 0.005
start = 5000
finish = 6000

audio_in = decode_mp3(fil)
samples = len(audio_in)
#print "intvl = ",intvl,' points = ',points
k = 0
pdata = numpy.zeros(samples)
sdata = numpy.zeros(samples)
ddata = numpy.zeros(samples)
for i in xrange(samples): pdata[i] = audio_in[i]
for i in xrange(samples): sdata[i] = gain*sin(2*pi*i*f[0]/sr + offset)
for i in xrange(samples): ddata[i] = pdata[i] - sdata[i]

plt.plot(pdata[start:finish], 'r.-')
plt.hold(True)
plt.plot(sdata[start:finish], 'b')
plt.plot(ddata[start:finish], 'g')
plt.grid(True)
#plt.show()
sar = numpy.array(32768.0*ddata, dtype=numpy.int16)
encode_mp3('diff.mp3',44100,1,128,sar)
plt.waitforbuttonpress(timeout=100.1)
        

