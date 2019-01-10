#!/usr/bin/env python
#coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np,matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from scipy import fft
from math import floor,log
import numpy
from scipy.io import wavfile

def save_fft(fil,audio_in):
    samples = len(audio_in)
    fft_size = 2**int(floor(log(samples)/log(2.0)))
    freq = fft(audio_in[0:fft_size])
    s_data = numpy.zeros(fft_size/2)
    x_data = numpy.zeros(fft_size/2)
    peak = 0;
    for j in xrange(fft_size/2):
        if (abs(freq[j]) > peak):
            peak = abs(freq[j])
            
    for j in xrange(fft_size/2):
        x_data[j] = log(2.0*(j+1.0)/fft_size);
        if (x_data[j] < -10):
            x_data[j] = -10
        s_data[j] = 10.0*log(abs(freq[j])/peak)/log(10.0)
    plt.ylim([-50,0])
    plt.plot(x_data,s_data)
    plt.title('fft log power')
    plt.grid()
    
    fields = fil.split('.')
    plt.savefig(fields[0]+'_fft.png', bbox_inches="tight")
    plt.clf()
    plt.close()
 

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs

""" plot spectrogram"""
def plotstft(samples, samplerate, binsize=2**10, plotpath=None, colormap="jet"):
    s = stft(samples, binsize)
    
    sshow, freq = logscale_spec(s, factor=16.0, sr=samplerate)
    ims = 20.*np.log10(1e-12+np.abs(sshow)/10e-6) # amplitude to decibel
    
    timebins, freqbins = np.shape(ims)
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()
        
    plt.clf()
    plt.close()
    
if __name__ == "__main__":
    (sr,inp) = wavfile.read("seem.wav")
    plotstft(inp,sr,binsize=1024,plotpath='seem.png')
