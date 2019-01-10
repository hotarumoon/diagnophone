#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import urllib
import urllib2
import re
import sys,os
import codecs
from scipy.io.wavfile import read,write
from scipy import signal
import numpy

mp = re.compile('\.mp3')

files = []
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
else:
    files = os.listdir('.')

count = 0
for fil in files:
    if (mp.search(fil)):
        (root,ext) = fil.split('.')
        cmd1 = 'sox \"'+fil+'\" -c 1 -t wav - | wav2png -w 300 -h 150 -b 2e4562ff -f ffb400aa -o '+root+'.png /dev/stdin'
        #print "cmd: ",cmd1
        os.system(cmd1)
        count = count+1
        print "saving png "+fil+" #"+str(count)
        #sys.exit(1)




