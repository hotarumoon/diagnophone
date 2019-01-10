#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re
import sys,os
import codecs
from math import sqrt,log
from scipy.io.wavfile import read,write
from scipy import signal
import numpy
import matplotlib
import pylab
import pysrt

show_plot = False
if (len(sys.argv) > 1):
    fil = sys.argv[1]

(root,ext) = fil.split('.')

subfile = root+".fa.srt"


try:
    subs = pysrt.open(subfile, encoding ='utf-8')
except:
    subs = pysrt.open(subfile, encoding ='iso-8859-1')

count = 0

times = []
sentences = []
for i in subs:
    s = subs[count]
    #t = int(44100*(60*s.start.minutes + s.start.seconds + 0.001*s.start.milliseconds))
    t = (60.0*s.start.minutes + s.start.seconds + 0.001*s.start.milliseconds)
    times.append(t)
    sentences.append(s.text)
    count = count+1
    #print s.text,t


Gap = 8
count = 0
# read audio samples
input_data = read(fil)
audio_in = input_data[1]

samples = len(audio_in)
cut_off = -28

print "samples = ",samples
seg = 4096
intvl = samples/seg
k = 0

data_out = []
cut2_count = 0
cut_count = 0

file_count = 0
save_count = 0
sample_count = 0

PB = open(root+"_for_db.txt",'w')


for i in xrange(intvl):
    sum = 0.0
    buffer_out = []
    for j in xrange(seg):
        s = float(audio_in[k][0])
        sum += (s*s)
        buffer_out.append(audio_in[k])
        k = k+1
    rms = sqrt(sum/seg)/16384.0
    if (rms == 0):
        rms_db = -100
    else:
        rms_db = 20.0*log(rms)/log(10.0)

    for samp in buffer_out:
        data_out.append(samp)
 
    cut2_count = cut2_count+1
    sample_count = sample_count + seg
    if (rms_db < cut_off):
        cut_count = cut_count+1
    else:
        cut_count = 0

    if (rms_db >= cut_off):
        if (save_count == 0):
            save_count = sample_count

    
    if (cut_count > Gap):
        if (cut2_count == Gap+1):
            pass
        else:
            if (save_count/44100.0 > times[file_count]):
                print sample_count/44100.0,save_count/44100.0,times[file_count],file_count
                #print sample_count/44100.0,4096*cut2_count/44100.0,file_count,times[file_count],sentences[file_count]
                sar = numpy.array(data_out, dtype=numpy.int16)
                fcount = '%04d' % file_count
                
                s = sentences[file_count]
                fname = fcount+'_'+s+'_'+str(times[file_count])+'.wav'
                
                #write(fname,44100,sar)
                s_out = str(file_count)+":"+s+":"+str(times[file_count])+":"+str(sample_count)+"\n"
                PB.write(s_out.encode('utf-8'))

                file_count = file_count+1
                data_out = []
                #if (file_count == 10): sys.exit(1)
        save_count = 0
        cut_count = 0
        cut2_count = 0

print 4096*cut2_count/44100.0,file_count
sar = numpy.array(data_out, dtype=numpy.int16)
fcount = '%04d' % file_count
fname = root+'_'+fcount+'.wav'
#write(fname,44100,sar)
s_out = str(file_count)+":"+s+":"+str(times[file_count])+":"+str(sample_count)+"\n"
PB.write(s_out.encode('utf-8'))

PB.close()




