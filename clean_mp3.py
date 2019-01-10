#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import codecs
#from scipy.io.wavfile import read,write
from scipy import signal
from lame  import *
import numpy

debug = False
if (len(sys.argv) > 1):
    fil = sys.argv[1]
else:
    sys.exit(1)

# read audio samples
audio_in = decode_mp3(fil)
#

b_hpf, a_hpf = signal.butter(6, 100/44100.0, 'high')
b_lpf, a_lpf = signal.butter(6, 8000/44100.0, 'low')
#w, h = signal.freqs(b, a)

# Approx 50 hz notch at 44.1khz
width = 0.995
center = -0.99997

az = numpy.array([1, 2.0*center, 1]);
a_notch = numpy.array([1, center*(1+width), width]);
b_notch = 0.5*(1+width)*az;

samples = len(audio_in)

if (samples > 10000):
    print "samples = ",samples,

    audio_lpf = signal.filtfilt(b_lpf,a_lpf,audio_in)
    audio_notch = signal.filtfilt(b_notch,a_notch,audio_in)
    audio = signal.filtfilt(b_hpf,a_hpf,audio_notch)

    max_v = 0
    for i in xrange(samples):
        if (abs(audio[i])>max_v): 
            max_v = abs(audio[i])

    gain = 0.8*32768.0/max_v
    print "gain = ",gain,

    data_out = []
    for i in xrange(samples):
        s = int(gain*audio[i])
        data_out.append(s)

    sar = numpy.array(data_out, dtype=numpy.int16)

    if ((gain > 1.1) or (gain < 0.9)):
        encode_mp3(fil,44100,1,128,sar)
else:
    cmd = 'rm \"'+fil+"\""
    print "cmd: ",cmd
    os.system(cmd)





