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
            plt.plot(xdata,pdata, 'r.')
            plt.hold(True)
            plt.grid(True)
            plt.title('fft log power over interval '+str(i))
            plt.pause(0.1)

        plt.hold(False)
        plt.waitforbuttonpress(timeout=2.1)

        log_sum_data = numpy.zeros(points)
        for j in xrange(points):
            log_sum_data[j] = 10.0*log(sum_data[j]/intvl)/log(10.0)
            sum_data[j] = sqrt(sum_data[j]/intvl)
                
        plt.plot(xdata,log_sum_data)
        plt.title('summed fft log power over all intervals')
        plt.draw()
        plt.waitforbuttonpress(timeout=2.1)
        plt.hold(True)
        
        k = 0
        var_data = numpy.zeros(points)
        for i in xrange(intvl):
            buffer_out = []
            for j in xrange(seg):
                buffer_out.append(audio_in[k])
                k = k+1
            buffer_out = abs(fft(buffer_out))
            for j in xrange(seg/2):
                if (j < points):
                    var_data[j] = var_data[j] + (buffer_out[j] - sum_data[j])*(buffer_out[j] - sum_data[j])/sum_data[j];

        log_var_data = numpy.zeros(points)
        for j in xrange(points):
            log_var_data[j] = 10.0*log(var_data[j]/intvl)/log(10.0)

        plt.plot(xdata,log_var_data,'g')
        plt.grid(True)
        plt.title('Log of Variance of fft power')
        plt.draw()
        plt.waitforbuttonpress(timeout=10.1)




