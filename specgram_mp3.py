#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import numpy
from scipy import signal
from pylab import *
from lame  import *

debug = False

if (len(sys.argv) > 1):
    fil = sys.argv[1]
else:
    sys.exit(1)

sig = decode_mp3(fil)
specgram(sig)
show()




