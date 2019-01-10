#!//Users/tkirke/anaconda/bin/python
import pysrt, sys, os

fil = sys.argv[1]
extra_proc = False
if (len(sys.argv) > 2):
    extra_proc = True
try:
    subs = pysrt.open(fil, encoding ='utf-8')
except:
    subs = pysrt.open(fil, encoding ='iso-8859-1')

size = len(subs)
count = 0

fields = fil.split('.')
PB = open(fields[0]+"_labels."+fields[1]+".txt",'w')
for i in subs:
    s = subs[count]
    count = count+1
    t = 60*s.start.minutes + s.start.seconds + 0.001*s.start.milliseconds
    txt = s.text
    txt = txt.replace('\n',' ')
    if (txt[0].isupper()):
        st = str(t)+"\t"+txt
        PB.write(st.encode('utf-8'))
PB.close()
