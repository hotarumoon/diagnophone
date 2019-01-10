#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import codecs
from math import sqrt,log
#from scipy.io.wavfile import read,write
from scipy import signal
from lame import *
import numpy
import matplotlib
import pylab

# Remove chunks more -27 db down from peak to remove audio 'gaps'
# optional plot envelope

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
for fil in files:
    if (mp.search(fil)):
        sig = decode_mp3(fil)
        samples = len(sig)
        seg = 1024
        intvl = samples/seg
        k = 0
        data_out = []
        plot_data_out = []
        plot_ref = []
        minsig = 0
        for i in xrange(intvl):
            sum = 0.0
            buffer_out = []
            for j in xrange(seg):
                s = float(sig[k])
                sum += (s*s)
                buffer_out.append(sig[k])
                k = k+1
            rms = sqrt(sum/seg)/16384.0
            rms_db = 20.0*log(rms)/log(10.0)
            plot_data_out.append(rms_db)
            for samp in buffer_out:
                data_out.append(samp)

        pdata = numpy.array(plot_data_out, dtype=numpy.float)
        pylab.plot(pdata)
        #pylab.ylim(-40,0)
        pylab.grid(True)
        pylab.show()

        #sar = numpy.array(data_out, dtype=numpy.int16)
        #write("junk.wav",44100,sar)




