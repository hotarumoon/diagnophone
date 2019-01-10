import re
import sys, os
import codecs
from math import sqrt, log
from scipy.io.wavfile import read, write
from scipy import signal, fft, ifft
from lame import *
import numpy
import matplotlib
import pylab

start_point = 0

seg = 1024 * 4


def get_noise(start):
    # read audio samples
    input_data = read('junk.wav')
    audio_in = input_data[1]
    samples = len(audio_in)
    intvl = (samples - start) / seg
    k = start
    sum_data = numpy.zeros(seg)
    for i in xrange(intvl):
        buffer_data = []
        for j in xrange(seg):
            buffer_data.append(audio_in[k])
            k = k + 1
        cbuffer_out = fft(buffer_data)
        for j in xrange(seg):
            sq = abs(cbuffer_out[j]) ** 2.0
            sum_data[j] = sum_data[j] + sq

    for j in xrange(seg):
        sum_data[j] = sqrt(sum_data[j] / intvl)
    return sum_data


def process_data(start, sum_data):
    input_data = read('junk.wav')
    audio_in = input_data[1]
    samples = len(audio_in)
    intvl = start / seg
    k = 0
    var_thres = 2.2
    data_out = []

    # print "intvl = ",intvl,start,seg
    for i in xrange(intvl):
        buffer_out = []
        for j in xrange(seg):
            buffer_out.append(audio_in[k])
            k = k + 1
        cbuffer_out = fft(buffer_out)
        for j in xrange(seg):
            if (abs(cbuffer_out[j]) < var_thres * sum_data[j]):
                cbuffer_out[j] = 0.02 * cbuffer_out[j];
        buf_out = ifft(cbuffer_out)
        for j in xrange(seg):
            data_out.append(buf_out[j].real)

    sar = numpy.array(data_out, dtype=numpy.int16)
    write("junk_out.wav", 44100, sar)
    cmd4 = 'lame junk_out.wav junk_out.mp3 >enc.log 2>&1 '
    os.system(cmd4)


def display_wav(filename):
    input_data = read(filename)
    audio_in = input_data[1]
    samples = len(audio_in)
    fig = pylab.figure();
    print
    samples / 44100.0, " seconds"
    k = 0
    plot_data_out = []
    for i in xrange(samples):
        plot_data_out.append(audio_in[k] / 32768.0)
        k = k + 1
    pdata = numpy.array(plot_data_out, dtype=numpy.float)
    pylab.plot(pdata)
    pylab.grid(True)
    pylab.ion()
    pylab.show()


def plot_sum_data(sum_data):
    pdata = numpy.array(sum_data, dtype=numpy.int16)
    pylab.figure()
    pylab.plot(pdata)
    pylab.grid(True)
    pylab.show()


def onclick(event):
    # print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
    start_point = int(event.xdata)
    sum_data = get_noise(start_point)
    # plot_sum_data(sum_data)
    process_data(start_point, sum_data)
    display_wav('junk_out.wav')
    cmd5 = 'play junk_out.mp3'
    os.system(cmd5)
    pylab.pause(2 ** 31 - 1)


mp = re.compile('\.mp3')

files = []
show_plot = False
if (len(sys.argv) > 1):
    files.append(sys.argv[1])
    if (len(sys.argv) > 2): show_plot = True
else:
    files = os.listdir('.')


def to_ascii_hex(fword):
    res = ""
    for c in fword:
        res += "%02x" % ord(c)
    return res


debug = False
count = 0
for fil in files:
    if (mp.search(fil)):
        # read audio samples
        audio_in = decode_mp3(fil)
        samples = len(audio_in)
        fig = pylab.figure();
        print
        samples / 44100.0, " seconds"
        seg = 1024
        intvl = samples / seg
        k = 0
        plot_data_out = []
        for i in xrange(intvl):
            for j in xrange(seg):
                plot_data_out.append(audio_in[k] / 32768.0)
                k = k + 1

        pdata = numpy.array(plot_data_out, dtype=numpy.float)
        pylab.plot(pdata)
        pylab.grid(True)
        fig.canvas.mpl_connect('button_press_event', onclick)
        pylab.show()