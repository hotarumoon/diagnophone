#!//Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Generate wav2png PNGs for all mp3 files in a directory
"""
import re,sys,os,codecs
import Queue
import threading
import time

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
    def run(self):
        print "Starting " + self.name
        process_data(self.name, self.q)
        print "Exiting " + self.name

def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            os.system(data)
            queueLock.release()
            print "%s processing %s" % (threadName, data)
        else:
            queueLock.release()
        time.sleep(1)

threadList = ["Thread-1", "Thread-2", "Thread-3", "Thread-4"]
queueLock = threading.Lock()
workQueue = Queue.Queue(10)
threads = []
threadID = 1

mp = re.compile('\.wav|\.mp3')

files = []
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
else:
    files = os.listdir('.')

# Create new threads
for tName in threadList:
    thread = myThread(threadID, tName, workQueue)
    thread.start()
    threads.append(thread)
    threadID += 1
    
queueLock.acquire()

count = 0
for fil in files:
    if (mp.search(fil)):
        (root,ext) = fil.split('.')
        if (ext == 'mp3'):
            cmd1 = 'sox \"'+fil+'\" -c 1 -t wav - | wav2png -w 800 -h 150 -b 2e4562ff -f ffb400aa -o \"'+root+'.png\" /dev/stdin >&/dev/null'
        else:
            cmd1 = 'wav2png '+fil+' -w 800 -h 150 -b 2e4562ff -f ffb400aa -o '+root+'.png  >&/dev/null'
        #os.system(cmd1)
        workQueue.put(cmd1)

queueLock.release()

# Wait for queue to empty
while not workQueue.empty():
    pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
    t.join()
print "Exiting Main Thread"
