#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re
import sys,os
import codecs
from math import sqrt,log
from scipy.io.wavfile import read,write
from scipy import signal
import numpy
import matplotlib
import pylab
from lame import *

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
PB = open('mp3_levels.txt','w')

count = 0
for fil in files:
    if (mp.search(fil)):
        audio_in = decode_mp3(fil)
        samples = len(audio_in)
        seg = 1024
        intvl = samples/seg
        k = 0
        minsig = 0
        for i in xrange(intvl):
            sum = 0.0
            for j in xrange(seg):
                s = float(audio_in[k])
                sum += (s*s)
                k = k+1
            rms = sqrt(sum/seg)/16384.0
            if (rms > 0): rms_db = 20.0*log(rms)/log(10.0)
            if (rms_db < minsig):
                minsig = rms_db
        db10 = '%02d' % int(-minsig)
        if (minsig > -20):
            s = "Minimum level is -"+db10+" dB in "+str(seg)+" sample segments over "+str(0.1*int(samples/4410))+" seconds for "+fil
            PB.write(s+"\n")
            cmd = 'mv \"'+fil+"\" ./levels/"
            os.system(cmd)
            print s

PB.close()



