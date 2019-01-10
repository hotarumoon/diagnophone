#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import codecs
from lame import *
import numpy

debug = False
if (len(sys.argv) > 1):
    fil = sys.argv[1]
else:
    sys.exit(1)

audio = decode_mp3(fil)
samples = len(audio)

if (samples > 10000):
    av = 0
    for i in xrange(samples):
        av += audio[i]
    av = av/samples

    max_v = 0
    for i in xrange(samples):
        if (abs(audio[i]-av)>max_v): 
            max_v = abs(audio[i]-av)

    gain = 0.8/max_v

    data_out = []
    for i in xrange(samples):
        s = int(gain*32768.0*audio[i] - av)
        data_out.append(s)

    print "samples = ",samples, "average = ",av, "gain = ",gain,

    sar = numpy.array(data_out, dtype=numpy.int16)
    if ((gain > 1.1) or (gain < 0.9)):
        encode_mp3(fil,44100,1,128,sar)
else:
    cmd = 'rm \"'+fil+"\""
    print "cmd: ",cmd
    os.system(cmd)





