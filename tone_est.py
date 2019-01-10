#!/Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
from math import sqrt,log,pi,sin,cos,atan2,floor
import cmath 
from scipy import signal,fft
import numpy

debug_estimates = False

# Quinn's method in
# B. G. Quinn, "Estimating Frequency by Interpolation Using Fourier
# Coefficients," IEEE Trans. Signal Processing, Vol. 42, no. 5, 1994.
#
# For Phase & Amplitude use Eq (9) from 
# B. G. Quinn, "Estimation of Frequency, Amplitude, and Phase from the
# DFT of a Time Series," IEEE Trans. Signal Processing, Vol. 45, no. 3,
# 1997
def c(k,d):
    cm1 = (cmath.exp(2*pi*1j*d)-1.0)/(4*pi*1j*(d-k))
    return cm1

def h(x):
    res = (0.25*log(3*x*x+6*x+1) - sqrt(6.0)*log(x+1-sqrt(2./3.))/24.0)/(x+1+sqrt(2./3.))
    return res
                                                                         
def k(x):
    res = 0.25*log(3*x*x+6*x+1) - sqrt(6.0)*log((x+1-sqrt(2./3.))/(x+1+sqrt(2./3.)))/24.0
    return res

def tone_est(sdata,sr):
    samples = len(sdata)
    fft_size = 2**int(floor(log(samples)/log(2.0)))
    freq = fft(sdata[0:fft_size])
    pdata = numpy.zeros(fft_size)
    for i in xrange(fft_size): pdata[i] = abs(freq[i])
    peak = 0
    peak_index = 0
    for i in xrange(fft_size/2):
        if (pdata[i] > peak):
            peak = pdata[i]
            peak_index = i

    R = peak*peak;
    p = (freq[peak_index+1].real * freq[peak_index].real + freq[peak_index+1].imag * freq[peak_index].imag)/R
    q = (freq[peak_index-1].real * freq[peak_index].real + freq[peak_index-1].imag * freq[peak_index].imag)/R

    g = -p/(1.0-p)
    e = q/(1.0-q)
    
    if ((p>0) and (q>0)):
        d = p
    else:
        d = q

    u = peak_index + d
    freq_est = u*sr/fft_size
    if (debug_estimates):
        print "peak is at ",peak_index,"(",u,") and is ",peak

    #d = 0.5*(p+q) + h(p*p) + h(q*q)
    #print "other peak index (2)", u+d

    sum_phase = freq[peak_index-1]*c(-1,d) + freq[peak_index]*c(0,d) + freq[peak_index+1]*c(1,d)
    sum_c_sq = abs(c(-1,d))*abs(c(-1,d)) + abs(c(0,d))*abs(c(0,d)) + abs(c(1,d))*abs(c(1,d))
    amp = (abs(sum_phase)/sum_c_sq)/fft_size
    phase_r = cmath.phase(sum_phase)

    return (amp,freq_est,phase_r)



def tone_est_near_index(sdata,index,range,sr):
    samples = len(sdata)
    fft_size = 2**int(floor(log(samples)/log(2.0)))
    freq = fft(sdata[0:fft_size])
    pdata = numpy.zeros(fft_size)
    for i in xrange(fft_size): pdata[i] = abs(freq[i])
    peak = 0
    peak_index = 0
    if (range == 0):
        peak = pdata[index]
        peak_index = index;
    else:
        for i in xrange(2*range):
            if (pdata[index+i-range] > peak):
                peak = pdata[index+i-range]
                peak_index = index+i-range


    R = peak*peak;
    p = (freq[peak_index+1].real * freq[peak_index].real + freq[peak_index+1].imag * freq[peak_index].imag)/R

    g = -p/(1.0-p)
    q = (freq[peak_index-1].real * freq[peak_index].real + freq[peak_index-1].imag * freq[peak_index].imag)/R
    e = q/(1.0-q)
    
    if ((p>0) and (q>0)):
        d = p
    else:
        d = q

    u = peak_index + d
    if (debug_estimates):
        print "peak is at ",peak_index,"(",u,") and is ",peak        

    sum_phase = freq[peak_index-1]*c(-1,d) +    freq[peak_index]*c(0,d) + freq[peak_index+1]*c(1,d)

    sum_c_sq = abs(c(-1,d))*abs(c(-1,d)) +    abs(c(0,d))*abs(c(0,d)) + abs(c(1,d))*abs(c(1,d))

    amp = (abs(sum_phase)/sum_c_sq)/fft_size

    phase_r = cmath.phase(sum_phase)
    freq_est = u*sr/fft_size

    return (amp,freq_est,phase_r)


def tone_est_above_index(sdata,index,sr):
    samples = len(sdata)
    fft_size = 2**int(floor(log(samples)/log(2.0)))
    freq = fft(sdata[0:fft_size])
    pdata = numpy.zeros(fft_size)
    for i in xrange(fft_size): pdata[i] = abs(freq[i])
    peak = 0
    peak_index = 0
    for i in xrange(fft_size/2):
        if (i > index):
            if (pdata[i] > peak):
                peak = pdata[i]
                peak_index = i

    R = peak*peak;
    p = (freq[peak_index+1].real * freq[peak_index].real + freq[peak_index+1].imag * freq[peak_index].imag)/R

    g = -p/(1.0-p)
    q = (freq[peak_index-1].real * freq[peak_index].real + freq[peak_index-1].imag * freq[peak_index].imag)/R
    e = q/(1.0-q)
    
    if ((p>0) and (q>0)):
        d = p
    else:
        d = q

    u = peak_index + d
    if (debug_estimates):
        print "peak is at ",peak_index,"(",u,") and is ",peak        

    sum_phase = freq[peak_index-1]*c(-1,d) +    freq[peak_index]*c(0,d) + freq[peak_index+1]*c(1,d)

    sum_c_sq = abs(c(-1,d))*abs(c(-1,d)) +    abs(c(0,d))*abs(c(0,d)) + abs(c(1,d))*abs(c(1,d))

    amp = (abs(sum_phase)/sum_c_sq)/fft_size

    phase_r = cmath.phase(sum_phase)
    freq_est = u*sr/fft_size

    return (amp,freq_est,phase_r)


def find_top_two_peaks(sdata):
    samples = len(sdata)
    fft_size = 2**int(floor(log(samples)/log(2.0)))
    freq = fft(sdata[0:fft_size])
    pdata = numpy.zeros(fft_size)
    for i in xrange(fft_size): pdata[i] = abs(freq[i])
    peak = 0
    peak1 = 0
    peak2 = 0
    peak1_index = 0
    peak2_index = 0
    for i in xrange(fft_size/2):
        if (pdata[i] > peak1):
            peak1 = pdata[i]
            peak1_index = i
    for i in xrange(fft_size/2):
        if (pdata[i] > peak2) and (abs(i - peak1_index) > 4):
            peak2 = pdata[i]
            peak2_index = i
    return (peak1,peak1_index,peak2,peak2_index)


# REMOVAL CASES

def old_est_tone_phase(sdata,a,f,sr):
    samples = len(sdata)
    points  = 360
    rms = numpy.zeros(points)
    sum_min = numpy.sum(numpy.square(sdata))
    min_index = 0
    for offset in xrange(points):
        sum = 0
        phase = pi*offset/180.0
        for i in xrange(samples):
            diff = (sdata[i] - a*cos(2*pi*i*f/sr + phase))
            sum += diff*diff
        rms[offset] = sum
        if (sum < sum_min):
            sum_min = sum
            min_index = offset
            #print "sum_min",sum_min,' index = ',min_index

    min_phase = pi*(min_index)/180.0
    #print "min for phase sweep is ",sum_min,' at offset ',min_index
    return min_phase


def find_min_phase(sdata,a,f,sr,phase):
    rms1 = 0
    rms2 = 0
    rms3 = 0
    samples = len(sdata)
    for i in xrange(samples):
        diff1 = (sdata[i] - a*cos(2*pi*i*f/sr + phase[0]))
        rms1 += diff1*diff1
        diff2 = (sdata[i] - a*cos(2*pi*i*f/sr + phase[1]))
        rms2 += diff2*diff2
        diff3 = (sdata[i] - a*cos(2*pi*i*f/sr + phase[2]))
        rms3 += diff3*diff3
    rms = numpy.zeros(3)
    rms[0] = rms1
    rms[1] = rms2
    rms[2] = rms3
    i = numpy.argmin(rms)
    p = phase[i]
    return i,p

        
def est_tone_phase(sdata,a,f,sr):
    delta = 120
    min_ang = 0.5
    p = 0
    phase = numpy.zeros(3)
    phase[0] = p+(-delta/180.0)*pi
    phase[1] = p
    phase[2] = p+(delta/180.0)*pi
    while (delta > min_ang):
        (i,p) = find_min_phase(sdata,a,f,sr,phase)
        delta = delta/2.0
        phase[0] = p+(-delta/180.0)*pi
        phase[1] = p
        phase[2] = p+(delta/180.0)*pi
        #print "p = ",(180.0*p/pi),'delta = ',delta
    
    min_phase = p
    #print "min for phase sweep is ",sum_min,' at offset ',min_index
    return min_phase


def est_tone_phase_and_remove(sdata,a,f,sr):
    samples = len(sdata)
    xdata = numpy.zeros(samples)
    min_phase = est_tone_phase(sdata,a,f,sr)
    for i in xrange(samples): xdata[i] = sdata[i] - a*cos(2*pi*i*f/sr + min_phase)
    return (xdata)

def tone_est_and_remove(sdata,sr,quant=False):
    (a,f,p) = tone_est(sdata,sr)
    if (quant):
        f = int(f+0.5)
    xdata = est_tone_phase_and_remove(sdata,a,f,sr)
    if (debug_estimates):
        print "removed sin with amplitude = ",a, " at frequency ",f
    return (xdata,f)

def tone_est_above_index_and_remove(sdata,index,sr):
    (a,f,p) = tone_est_above_index(sdata,index,sr)
    xdata = est_tone_phase_and_remove(sdata,a,f,sr)
    if (debug_estimates):
        print "removed sin with amplitude = ",a, " at frequency ",f
    return (xdata,f)

def tone_est_near_index_and_remove(sdata,index,range,sr,quant=False):
    (a,f,p) = tone_est_near_index(sdata,index,range,sr)
    if (quant): f = int(f+0.5)
    xdata = est_tone_phase_and_remove(sdata,a,f,sr)
    if (debug_estimates):
        print "removed sin with amplitude = ",a, " at frequency ",f
    return (xdata,f)

def remove_n_harmonics(audio_in,sr,n,f0):
    samples = len(audio_in)
    f_scale = f0*(2**int(floor(log(samples)/log(2.0))))/sr
    for i in xrange(n):
        hnum = 1.0+i
        new_index = int(floor(hnum*f_scale))
        (audio_in,f1) = tone_est_near_index_and_remove(audio_in,new_index,0,sr,True)
    return audio_in


