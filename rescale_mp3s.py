#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import urllib
import urllib2
import re
import sys,os
import codecs
from scipy.io.wavfile import read,write
from lame import *
import numpy


mp = re.compile('\.mp3')
files = os.listdir('.')

def to_ascii_hex(fword):
    res = ""
    for c in fword:
        res += "%02x" % ord(c)
    return res

# Rename to 'ascii' friendly , lame -> wav, octave -> rescale, lame -> .mp3
debug = False

num_files = len(files)
count = 0
for fil in files:
    if (mp.search(fil)):
        audio = decode_mp3(fil)
        mono = True
        samples = len(audio)

        ss = ''
        if (samples > 10000):
            ss = str(int(10.0*samples/44100)*0.1)+" secs "
            max_v = 0
            for i in xrange(samples):
                if (mono):
                    if (abs(audio[i])>max_v): max_v = abs(audio[i])
                else:
                    if (abs(audio[i][0])>max_v): max_v = abs(audio[i][0])

            gain = 0.99/max_v
            ss += "gain = "+str(int(100.0*gain)/100.0)+" "

            data_out = []
            for i in xrange(samples):
                if (mono):
                    s = int(gain*32768.0*audio[i])
                else:
                    s = int(gain*32768.0audio[i][0])

                data_out.append(s)

            sar = numpy.array(data_out, dtype=numpy.int16)
            if ((gain > 1.1) or (gain < 0.9)):
                encode_mp3(fil,44100,1,128,sar)
                print ss+"processing "+fil+" "+str(count)+"/"+str(num_files)
            else:
                print str(count)+"/"+str(num_files)
        else:
            cmd = 'mv \"'+fil+"\" ./tmp/"
            print "mv: ",cmd
            os.system(cmd)

        count = count+1




