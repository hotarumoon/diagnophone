#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import numpy
from scipy.io.wavfile import read,write
from scipy import signal
from pylab import *
from lame  import *

debug = False

if (len(sys.argv) > 1):
    fil = sys.argv[1]
else:
    sys.exit(1)

(root,ext) = fil.split('.')
print "Ext is ",ext
if (ext == 'mp3'):
    sig = decode_mp3(fil)
elif (ext == 'wav'):
    inp = read(fil)
    s = inp[1]
    #print "type(sig) = ",type(s),s.shape
    if (s.shape[0] != s.size):
        sig = s[:,0]+s[:,1]
    else:
        sig = s
else:
    print "unknown file type",fil
    sys.exit(1)
        
specgram(sig)
savefig(root+'.png')
show()




