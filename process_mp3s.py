#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import codecs
#from scipy.io.wavfile import read,write
from scipy import signal
from lame import *
import numpy

b_hpf, a_hpf = signal.butter(6, 100/44100.0, 'high')
b_lpf, a_lpf = signal.butter(6, 8000/44100.0, 'low')

wn = 50/44100.0
r = 0.995
a_notch = numpy.array([1, -2.0*r*numpy.cos(2*numpy.pi*wn), r*r]);
b_notch = numpy.array([1, -2.0*numpy.cos(2*numpy.pi*wn), 1.0]);

# Approx 50 hz notch at 44.1khz
width = 0.995
center = -0.99997 # -cos(2*pi*50/44100)
bz = numpy.array([1, 2.0*center, 1]);
a_notch = numpy.array([1, center*(1+width), width]);
b_notch = 0.5*(1+width)*bz;

mp = re.compile('\.mp3')

files = []
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
else:
    files = os.listdir('.')

debug = False

use_lpf = False

count = 0
for fil in files:
    if (mp.search(fil)):
        print "processing ",fil
        audio_in = decode_mp3(fil)
        samples = len(audio_in)
        #print len(b_lpf),len(a_lpf),len(audio_in)

        if (samples > 10000):
            if (use_lpf):
                audio_lpf   = signal.filtfilt(b_lpf,a_lpf,audio_in)
            else:
                audio_lpf   = audio_in
                
            audio_notch = signal.filtfilt(b_notch,a_notch,audio_lpf)
            audio       = signal.filtfilt(b_hpf,a_hpf,audio_notch)

            max_v = 0
            for i in xrange(samples):
                if (abs(audio[i])>max_v): 
                    max_v = abs(audio[i])

            gain = 0.8/max_v

            data_out = []
            for i in xrange(samples):
                s = int(gain*32768.0*audio[i])
                data_out.append(s)

            sar = numpy.array(data_out, dtype=numpy.int16)
            encode_mp3(fil,44100,1,128,sar)
        else:
            cmd = 'rm \"'+fil+"\""

        count = count+1
        dur = 0.1*floor(10.0*samples/44100)
        gain = 0.1*floor(10.0*gain)
        print " ",dur," seconds, with gain = ",gain,", file #",str(count)




