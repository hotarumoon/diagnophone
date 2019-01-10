#!//Users/tkirke/anaconda/bin/python
import pysrt, sys, os

fil = sys.argv[1]
try:
    subs = pysrt.open(fil, encoding ='utf-8')
except:
    subs = pysrt.open(fil, encoding ='iso-8859-1')

size = len(subs)
count = 0

fields  = fil.split('.')
PB = open(fields[0]+'.txt','w')
for i in subs:
    s = subs[count]
    count = count+1
    t = int(44100*(60*s.start.minutes + s.start.seconds + 0.001*s.start.milliseconds))
    txt = s.text
    txt = txt.replace('\n',' ')
    #print "txt[0] = ",txt[0],txt
    if (txt[0].isupper()):
        PB.write('\n')
    else:
        PB.write(' ')
    try:
        num = int(txt[0])
        PB.write('\n')
    except:
        pass
    PB.write(txt.encode('utf-8'))

PB.write('\n')
PB.close()
