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


def remove_harmonics(audio_in,sr,numh):
    samples = len(audio_in)
    (adata,f0) = tone_est_and_remove(audio_in,sr,True)
    f_scale = f0*(2**int(floor(log(samples)/log(2.0))))/sr
    for i in xrange(numh):
        hnum = 2.0+i
        print "removing ",hnum
        if (i<20):
            new_index = int(floor(hnum*f_scale))
            (adata,f1) = tone_est_near_index_and_remove(adata,new_index,4,sr,True)
        else:
            new_index = int(floor(20*f_scale))
            (adata,f1) = tone_est_above_index_and_remove(adata,new_index,sr)
    return adata


def remove_harmonics_from(audio_in,sr,f0,numh):
    samples = len(audio_in)
    adata = audio_in
    f_scale = f0*(2**int(floor(log(samples)/log(2.0))))/sr
    for i in xrange(numh):
        hnum = 1.0+i
        if (i<20):
            new_index = int(floor(hnum*f_scale))
            (adata,f1) = tone_est_near_index_and_remove(adata,new_index,4,sr,True)
        else:
            new_index = int(floor(20*f_scale))
            (adata,f1) = tone_est_above_index_and_remove(adata,new_index,sr)
        #print "removing ",hnum, ' at index,',new_index
    return adata

mp = re.compile('\.mp3')

files = []
show_plot = False
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
    if (len(sys.argv) > 2): show_plot = True
else:
    files = os.listdir('.')

points = 44100/50

eps = 1e-3 # -60 dB for plots
for fil in files:
    (sr,audio_in) = get_samples(fil)
    samples = len(audio_in)
    intvl = int(floor(samples/points))
    print "samples = ",samples,' interval = ',intvl, ' points = ',points
    #adata = remove_harmonics(audio_in,sr,30)
    adata = remove_harmonics_from(audio_in,sr,50.0,30)
    
    out_fil = "diff_h_"+fil
    gain = (2**15)
    save_samples(out_fil,44100,1,128,numpy.array(numpy.multiply(gain,adata),dtype=numpy.int16))

    plt.xlim([0,4000]);
    plt.plot(audio_in, 'r')
    plt.hold(True)
    plt.grid(True)
    plt.plot(adata, 'b')
    plt.show()

