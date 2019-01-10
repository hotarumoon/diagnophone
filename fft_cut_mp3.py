#!/usr/bin/python
# -*- coding: utf-8 -*-
import re, sys,os, codecs
from time import sleep
from math import sqrt,log
from scipy import signal,fft,ifft
from lame  import *
import numpy, matplotlib, pylab

# if audio data RMS is below a threshold * overall RMS at that frequency set audio data completely to 0 at those frequencies and do IFFT

mp = re.compile('\.mp3')


files = []
show_plot = False
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
    if (len(sys.argv) > 2): show_plot = True
else:
    files = os.listdir('.')

debug = False
var_thres = 2.2

count = 0
for fil in files:
    if (mp.search(fil)):
        audio_in = decode_mp3(fil)
        samples = len(audio_in)
        seg = 1024*4
        intvl = samples/seg
        k = 0
        sum_data = numpy.zeros(seg)
        for i in xrange(intvl):
            buffer_out = []
            for j in xrange(seg):
                buffer_out.append(audio_in[k])
                k = k+1
            buffer_out = (fft(buffer_out))
            for j in xrange(seg):
                sq = abs(buffer_out[j])**2.0
                sum_data[j] = sum_data[j]+sq

        # Get rms
        for j in xrange(seg):
            sum_data[j] = sqrt(sum_data[j]/intvl)

        k = 0
        data_out=[]
        for i in xrange(intvl):
            buffer_out = []
            for j in xrange(seg):
                buffer_out.append(audio_in[k])
                k = k+1
            cbuffer_out = fft(buffer_out)
            for j in xrange(seg):
                if (abs(cbuffer_out[j]) < var_thres*sum_data[j]):
                    cbuffer_out[j] = 0.02*cbuffer_out[j];

            buf_out = ifft(cbuffer_out)
            for j in xrange(seg):
                data_out.append(buf_out[j].real)

        sar = numpy.array(data_out, dtype=numpy.int16)
        encode_mp3(fil,44100,1,128,sar)
        print "processing "+fil+" "+str(count)




