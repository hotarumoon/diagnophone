#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import codecs
from lame import *
import numpy

def calc_average_and_max(audio):
    av = 0
    samples = len(audio)
    for i in xrange(samples):
        av += audio[i]
    av = av/samples

    max_v = 0
    for i in xrange(samples):
        if (abs(audio[i]-av)>max_v): 
            max_v = abs(audio[i]-av)

    return av,max_v

def rescale_audio_floats(audio):
    (av,max_v) = calc_average_and_max(audio)
    gain = 0.8/max_v
    samples = len(audio)
    for i in xrange(samples):
        audio[i] = (gain*audio[i] - av)
    return audio

def rescale_audio_ints(audio):
    (av,max_v) = calc_average_and_max(audio)
    gain = 32768.0*0.8/max_v
    samples = len(audio)
    for i in xrange(samples):
        audio[i] = int(gain*audio[i] - av)
    return audio

if __name__ == "__main__":
    fil = sys.argv[1]
    (sr,audio) = get_samples(fil)
    samples = len(audio)

    (av,max_v) = calc_average_and_max(audio)
    gain = 0.8/max_v
    data_out = []
    for i in xrange(samples):
        s = int(gain*32768.0*audio[i] - av)
        data_out.append(s)

    #print "samples = ",samples, "average = ",av, "gain = ",gain,

    if ((gain > 1.1) or (gain < 0.9)):
        sar = numpy.array(data_out, dtype=numpy.int16)
        save_samples(fil,sr,1,128,sar)





