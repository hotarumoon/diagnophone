#!/Users/tkirke/anaconda/bin/python
import time
import sys
import os
from lame  import *
import numpy

files = []
if (len(sys.argv) > 1):
    for s in sys.argv:
        files.append(s)

if __name__ == "__main__":
    sr = 0
    try:
        (sr,audio1) = get_samples(files[1])
        (sr,audio2) = get_samples(files[2])
    except:
        print "problem with "+files
    audio = audio1 - audio2;
    samples = len(audio1)
    # then convert from floating point to ints
    vals = numpy.array(numpy.multiply(2**15,audio),dtype=numpy.int16)
    #(root,ext) = files[1].split('.')
    save_samples('diff.wav',sr,1,128,vals)

