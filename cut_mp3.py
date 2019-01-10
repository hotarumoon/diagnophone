#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
import codecs
from math import sqrt,log
#from scipy.io.wavfile import read,write
from scipy import signal
from lame  import *
import numpy
import matplotlib
import pylab

# Remove chunks more -27 db down from peak to remove audio 'gaps'
# optional plot envelope

mp = re.compile('\.mp3')

files = []
show_plot = False
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
    if (len(sys.argv) > 2): show_plot = True
else:
    files = os.listdir('.')

debug = False

count = 0
for fil in files:
    if (mp.search(fil)):
        audio_in = decode_mp3(fil)
        mono = True
        samples = len(audio_in)

        cut_off = -28
        cut_count = 0

        print "samples = ",samples
        if (samples > 10000):
            seg = 4096
            intvl = samples/seg
            k = 0
            data_out = []
            plot_data_out = []
            plot_ref = []
            snipped = False
            for i in xrange(intvl):
                sum = 0.0
                buffer_out = []
                for j in xrange(seg):
                    if (mono):
                        s = float(audio_in[k])
                    else:
                        s = float(audio_in[k][0])
                    sum += (s*s)
                    buffer_out.append(audio_in[k])
                    k = k+1
                rms = sqrt(sum/seg)/16384.0
                rms_db = 20.0*log(rms)/log(10.0)
                plot_data_out.append(rms_db)
                plot_ref.append(cut_off)
                if (rms_db > cut_off):
                    for samp in buffer_out:
                        if (mono):
                            data_out.append(samp)
                        else:
                            data_out.append(samp[0])
                else:
                    cut_count = cut_count+1
                    snipped = True
                    
            if (show_plot):
                pdata = numpy.array(plot_data_out, dtype=numpy.float)
                pdata_r = numpy.array(plot_ref, dtype=numpy.float)
                pylab.plot(pdata)
                pylab.plot(pdata_r)
                pylab.ylim(-40,0)
                pylab.grid(True)
                pylab.show()

            sar = numpy.array(data_out, dtype=numpy.int16)
            if (snipped):
                encode_mp3(fil,44100,1,128,sar)
                print int(10*cut_count*4096/44100.0)*0.1,' seconds cut'

            count = count+1
            print "processing "+fil+" "+str(count)



