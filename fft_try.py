#!/Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os,codecs
from time import sleep
from math import sqrt,log
from scipy import signal,fft
import numpy, matplotlib
from lame  import *
from tone_est import *
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

peak1 = 0
peak2 = 0

eps = 1e-3 # -60 dB for plots
for fil in files:
    if (mp.search(fil)):
        audio_in = decode_mp3(fil)
        samples = len(audio_in)
        seg = 1024*4
        points = 64*4096/seg
        intvl = samples/seg
        f_inc = (0.5*sr/seg)
        xdata = numpy.zeros(points)
        for j in xrange(points): xdata[j] = f_inc*j
        #print "intvl = ",intvl,' points = ',points
        k = 0

        fft_size = 2**int(floor(log(samples)/log(2.0)))
        (peak1,peak1_index,peak2,peak2_index) = find_top_two_peaks(audio_in)
        
        peak_diff = 20.0*log(peak1/peak2)/log(10.0)
        print "Peak diff = ",peak_diff, " dB"
        print "Freq diff = ",sr*(peak1_index-peak2_index)/fft_size

        if (peak_diff > 6) and (abs(peak1_index-peak2_index) > 4):
            (audio_in,f) = tone_est_and_remove(audio_in,sr)

        
        sum_data = numpy.zeros(points)
        for i in xrange(intvl):
            buffer_out = []
            for j in xrange(seg):
                buffer_out.append(audio_in[k])
                k = k+1
            buffer_out = abs(fft(buffer_out))+eps
            pdata = numpy.zeros(points)
            for j in xrange(seg/2):
                if (j < points):
                    sq = buffer_out[j]*buffer_out[j]
                    sum_data[j] = sum_data[j]+sq
                    pdata[j] = 20.0*log(buffer_out[j])/log(10.0);

        log_sum_data = numpy.zeros(points)
        for j in xrange(points):
            log_sum_data[j] = 10.0*log(sum_data[j]/intvl)/log(10.0)
                
        plt.plot(xdata,log_sum_data)
        plt.title('summed fft log power over all intervals')
        plt.draw()
        plt.waitforbuttonpress(timeout=12.1)
        plt.hold(True)

        




