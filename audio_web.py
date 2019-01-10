#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os,codecs
from scipy import signal
import numpy

"""
Generate html files for all mp3 files in a directory
with 25 PNGs in each html file with audio control to play audio for each mp3
"""


hdr = """
<meta charset="utf-8">
<!DOCTYPE html>
<html>
<body>
"""

hdr_end = """
</body>
</html>
"""
mp = re.compile('\.wav|\.mp3')

files = []
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
else:
    files = os.listdir('.')

count = 0
fcount = 0

rows = 25
for fil in files:
    if (mp.search(fil)):
        if (count % rows == 0):
            if (count > 0):
                PT.write("</ul>\n")
                PT.write(hdr_end)
                PT.close()
            fc = '%03d' % fcount
            PT = open('my'+fc+'.html','w')
            PT.write(hdr)
            fcount = fcount+1  
            fcp1 = '%03d' % fcount
            nex = "<a href=\"my"+fcp1+".html\">next</a><p>\n"
            PT.write(nex)
            nex2 = '<ul class="cols" rel="3">'
            PT.write(nex2)

        (root,ext) = fil.split('.')
        if (ext == 'mp3'):
            cmd1 = 'sox \"'+fil+'\" -c 1 -t wav - | wav2png -w 800 -h 150 -b 2e4562ff -f ffb400aa -o \"'+root+'.png\" /dev/stdin >&/dev/null'
        else:
            cmd1 = 'wav2png '+fil+' -w 800 -h 150 -b 2e4562ff -f ffb400aa -o '+root+'.png  >&/dev/null'
        #print cmd1,' ',count
        os.system(cmd1)
        text1 = '<li><audio controls>  <source src="'+fil+'" type="audio/mpeg"> </audio>'
        text1 = text1 + '<p>'+str(count)+" "+fil+'</p>'
        PT.write(text1+"\n")
        text2 = '<img src="'+root+'.png" style="width:800px;height:150px;">'
        PT.write(text2+"<p></p></li>\n")
        count  = count + 1

PT.write(hdr_end)
PT.close()



