#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re
import sys,os
import codecs
from math import sqrt,log
from scipy.io.wavfile import read,write
from scipy import signal
import numpy
import pysrt
import datetime
import json
import fileinput

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
    t = int(44100*(60.0*s.start.minutes + s.start.seconds + 0.001*s.start.milliseconds))
    times.append(t)
    sentences.append(s.text)
    count = count+1

times.append(2^31)
sentences.append(' ')
    #print s.text,t


# read audio samples
input_data = read(fil)
audio_in = input_data[1]
samples = len(audio_in)
#print "samples = ",samples

jf = open(root+".json",'w')
jf.write("{\"words\" : [\n")

use_eng = False
eng_file = root+'_labels.en.txt'
eng = []
if os.path.isfile(eng_file):
    use_eng = True
    for line in fileinput.input(eng_file):
        fields = line.split()
        words = ' '.join(fields[1:])
        eng.append(words)

k = 0
data_out = []
file_count = 0
eng_count = 0
dict = {}
even = True
for i in xrange(samples):
    #data_out.append(audio_in[k])
    k = k+1
    if ((file_count+1) < len(times)):
        if (k > times[file_count+1]):
            #sar = numpy.array(data_out, dtype=numpy.int16)
            fcount = '%04d' % file_count
            s = sentences[file_count]
            sec = int(times[file_count]/44100.0)
            ts = str(datetime.timedelta(seconds=sec))
            #print "sec",sec,ts
            fname = fcount+'_'+s+'_'+ts+'.wav'
            #write(fname,44100,sar)
            dict = {}
            dict['count'] = file_count
            dict['farsi'] = sentences[file_count]
            dict['time'] = ts
            dict['sample'] = k
            if (even):
                #print eng[eng_count]
                try:
                    dict['english'] = eng[eng_count].decode('utf-8')
                    eng_count = eng_count+1
                except:
                    dict['english'] = ''
            else:
                dict['english'] = ''
            even = not even
            json_data = json.dumps(dict,ensure_ascii=False)
            js = json_data.encode('utf-8')+","
            jf.write(js+"\n")
            file_count = file_count+1
            data_out = []

#sar = numpy.array(data_out, dtype=numpy.int16)
fcount = '%04d' % file_count
fname = root+'_'+fcount+'.wav'
#write(fname,44100,sar)
sec = int(times[file_count]/44100.0)
ts = str(datetime.timedelta(seconds=sec))

dict = {}
dict['count'] = file_count
dict['farsi'] = sentences[file_count]
dict['time'] = ts
dict['sample'] = k
if (even):
    try:
        dict['english'] = eng[eng_count]
    except:
        dict['english'] = ''
else:
    dict['english'] = ''
json_data = json.dumps(dict,ensure_ascii=False)
js = json_data.encode('utf-8')
jf.write(js+"\n")
jf.write("]\n}\n")
jf.close()






