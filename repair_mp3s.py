#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
# Run Mp3check on all mp3 files in a directory
import re,os

mp = re.compile('\.mp3')
files = os.listdir('.')

num_files = len(files)
count = 0
for fil in files:
    if (mp.search(fil)):
        cmd = 'mp3check -e --cut-junk-start \"'+fil+"\" >&tmp.log"
        print cmd," ",count,"/",num_files
        os.system(cmd)
        count = count+1




