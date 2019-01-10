#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import codecs
from scipy.io.wavfile import read,write
from scipy import signal
from lame  import *
import numpy

#%wn = f_cut/rate    
#%r = 0.99
#%B, A = np.zeros(3), np.zeros(3)
#%A[0],A[1],A[2] = 1.0, -2.0*r*np.cos(2*np.pi*wn), r*r
#%B[0],B[1],B[2] = 1.0, -2.0*np.cos(2*np.pi*wn), 1.0

# Approx 50 hz notch at 44.1khz
width = 0.995
center = -0.99997
bz = numpy.array([1, 2.0*center, 1]);
a_notch = numpy.array([1, center*(1+width), width]);
b_notch = 0.5*(1+width)*bz;
b_hpf, a_hpf = signal.butter(6, 100/44100.0, 'high')

mp = re.compile('\.mp3')

files = []
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
else:
    files = os.listdir('.')

debug = False
count = 0
for fil in files:
    if (mp.search(fil)):
        audio_in = decode_mp3(fil)
        samples = len(audio_in)
        #print len(b_lpf),len(a_lpf),len(audio_in)

        if (samples > 10000):
            audio_notch = signal.filtfilt(b_notch,a_notch,audio_in)
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
            if ((gain > 1.1) or (gain < 0.9)):
                encode_mp3(fil,44100,1,128,sar)
        else:
            cmd = 'rm \"'+fil+"\""

        count = count+1
        print "processing "+fil+" "+str(count),"samples = ",samples, "gain = ",gain




