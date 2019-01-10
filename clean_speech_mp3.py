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


def to_ascii_hex(fword):
    res = ""
    for c in fword:
        res += "%02x" % ord(c)
    return res

# Rename to 'ascii' friendly , lame -> wav, rescale, lame -> .mp3
debug = False

if (len(sys.argv) > 1):
    fil = sys.argv[1]
else:
    sys.exit(1)


new_name = to_ascii_hex(fil)
cmd1 = 'cp \"'+fil+'\" '+new_name+'.mp3'
if (debug): print "cmd: ",cmd1
os.system(cmd1)
cmd2 = 'lame --decode '+new_name+'.mp3 junk.wav >dec.log 2>&1 '
if (debug): print "cmd: ",cmd2
os.system(cmd2)

cmd3 = 'clean_speech junk.wav clean.wav'
if (debug): print "cmd: ",cmd3
os.system(cmd3)

cmd4 = 'lame clean.wav \"'+fil+'\" >enc.log 2>&1 '
if (debug): print "cmd:",cmd4
os.system(cmd4)

cmd5 = 'rm '+new_name+'.mp3'
if (debug): print "cmd:",cmd5
os.system(cmd5)




