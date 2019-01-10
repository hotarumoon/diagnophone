#!//Users/tkirke/anaconda/bin/python
import pysrt, sys, os

fil = sys.argv[1]
try:
    subs = pysrt.open(fil, encoding ='utf-8')
except:
    subs = pysrt.open(fil, encoding ='iso-8859-1')

size = len(subs)
count = 0
for i in subs:
    s = subs[count]
    count = count+1
    t = (44100*(60*s.start.minutes + s.start.seconds + 0.001*s.start.milliseconds))
    print str(t)+"\t"+s.text
