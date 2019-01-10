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

def plot_and_wait(audio_in,d,col):
    print "starting plot"
    plt.xlim([0,4000]);
    plt.ylim([-1,1]);
    plt.plot(audio_in, 'r')
    plt.grid(True)
    plt.hold(True)
    plt.plot(d,col)
    plt.hold(False)
    plt.draw()

def plot_fft(d):
    eps = 1e-6
    d = abs(fft(d))+eps
    plt.plot(d,'r')
    plt.grid(True)
    plt.hold(False)
    plt.xlim([0,400]);
    plt.draw()

for i in xrange(samples):
    audio_in[i] = sin(2*pi*f0*i+phase1) - 0.5*sin(2*pi*2*f0*i+phase2) + 0.25*sin(2*pi*3*f0*i+phase3) - 0.125*sin(2*pi*4*f0*i+phase4) + 0.0625*sin(2*pi*5*f0*i+phase5);


intvl = int(floor(samples/points))
print "samples = ",samples,' interval = ',intvl, ' points = ',points

if (show_t):
    plt.xlim([0,4000]);
    plt.plot(audio_in, 'r')
    plt.grid(True)
    plt.show()
    sys.exit(1)

plot_and_wait(audio_in,audio_in,'r')
#plot_fft(audio_in)
plt.waitforbuttonpress(timeout=2.1)

(adata,f0) = tone_est_and_remove(audio_in,sr,True)
fft_size = 2**int(floor(log(samples)/log(2.0)))
new_index = int(floor(2.0*f0*fft_size/sr))
plot_and_wait(audio_in,adata,'b')

#plot_fft(adata)
plt.waitforbuttonpress(timeout=2.1)
#print "removed f0 = ",f0, ' start near index = ', new_index
(bdata,f1) = tone_est_near_index_and_remove(adata,new_index,4,sr,True)
plot_and_wait(audio_in,bdata,'g')
#plot_fft(bdata)
plt.waitforbuttonpress(timeout=2.1)

new_index = int(floor(4.0*f0*fft_size/sr))
#print "removed 2*f0 = ",f1, ' start near index = ', new_index
(cdata,f2) = tone_est_near_index_and_remove(bdata,new_index,4,sr,True)
#plot_fft(cdata)
plot_and_wait(audio_in,cdata,'k')
plt.waitforbuttonpress(timeout=2.1)
new_index = int(floor(3.0*f0*fft_size/sr))
#print "removed 4*f0 = ",f2,' start near index = ', new_index
(ddata,f3) = tone_est_near_index_and_remove(cdata,new_index,4,sr,True)
#plot_fft(ddata)
plot_and_wait(audio_in,ddata,'y')
plt.waitforbuttonpress(timeout=2.1)
new_index = int(floor(5.0*f0*fft_size/sr))
#print "removed 3*f0 = ",f3,' start near index = ', new_index
(edata,f4) = tone_est_near_index_and_remove(ddata,new_index,4,sr,True)
#plot_fft(edata)
plot_and_wait(audio_in,edata,'b')
plt.waitforbuttonpress(timeout=2.1)
#print "removed 5*f0 = ",f4
new_index = int(floor(f0*fft_size/sr))
print "removed 5*f0 = ",f3,' start near index = ', new_index
(fdata,f5) = tone_est_near_index_and_remove(edata,new_index,4,sr,True)
#plot_fft(fdata)
plot_and_wait(audio_in,fdata,'g')
plt.show()
print "removed f0 = ",f5

if (save_files):    
    out_fil = "diff_"+fil
    gain = (2**15)
    save_samples(out_fil,44100,1,128,numpy.array(numpy.multiply(gain,xdata),dtype=numpy.int16))
    out_fil = "diff3_"+fil
    save_samples(out_fil,44100,1,128,numpy.array(numpy.multiply(gain,zdata),dtype=numpy.int16))
    out_fil = "diff4_"+fil
    save_samples(out_fil,44100,1,128,numpy.array(numpy.multiply(gain,adata),dtype=numpy.int16))
    out_fil = "diff5_"+fil
    save_samples(out_fil,44100,1,128,numpy.array(numpy.multiply(gain,bdata),dtype=numpy.int16))
    out_fil = "diff6_"+fil
    save_samples(out_fil,44100,1,128,numpy.array(numpy.multiply(gain,cdata),dtype=numpy.int16))

#plt.plot(cdata, 'y')
#plt.waitforbuttonpress(timeout=100.1)

