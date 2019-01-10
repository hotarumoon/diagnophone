#!//Users/tkirke/anaconda/bin/python
# Make sure there are no gaps in presenting subtitles!
import pysrt, sys, os

files = os.listdir('.')

for fil in files:
    print fil
    try:
        subs = pysrt.open(fil, encoding ='utf-8')
    except:
        subs = pysrt.open(fil, encoding ='iso-8859-1')

    size = len(subs)
    count = 0
    for i in subs:
        if (count < size-1):
            subs[count].end = subs[count+1].start
        count = count+1

    fileout = fil+'.txt'
    subs.save(fileout,encoding = 'utf-8')
