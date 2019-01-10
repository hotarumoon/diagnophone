#!/Users/tkirke/anaconda/bin/python
# -*- coding: utf-8 -*-
import re,sys,os
from math import sqrt,log,pi,sin,cos,atan2,floor
import cmath 
from scipy import signal,fft
import numpy

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

    g = -p/(1.0-p)
    q = (freq[peak_index-1].real * freq[peak_index].real + freq[peak_index-1].imag * freq[peak_index].imag)/R
    e = q/(1.0-q)
    
    if ((p>0) and (q>0)):
        d = p
    else:
        d = q

    u = peak_index + d
    print "peak is at ",peak_index,"(",u,") and is ",peak        
    #print "u = ",0.5*u*sr/fft_size,' f[0] = ',f[0]

    sum_phase = freq[peak_index-1]*c(-1,d) + freq[peak_index]*c(0,d) + freq[peak_index+1]*c(1,d)

    sum_c_sq = abs(c(-1,d))*abs(c(-1,d)) + abs(c(0,d))*abs(c(0,d)) + abs(c(1,d))*abs(c(1,d))

    amp = (abs(sum_phase)/sum_c_sq)/fft_size

    phase_r = cmath.phase(sum_phase)
    freq_est = 0.5*u*sr/fft_size


    return (amp,freq_est,phase_r)


def est_tone_phase(sdata,a,f,sr):
    samples = len(sdata)
    points  = 360
    rms = numpy.zeros(points)
    sum_min = numpy.sum(numpy.square(sdata))
    min_index = 0
    for offset in xrange(points):
        sum = 0
        phase = pi*offset/180.0
        for i in xrange(samples):
            diff = (sdata[i] - a*cos(2*pi*i*f/(sr/2.0) + phase))
            sum += diff*diff
        rms[offset] = sum
        if (sum < sum_min):
            sum_min = sum
            min_index = offset
            #print "sum_min",sum_min,' index = ',min_index

    min_phase = pi*(min_index)/180.0
    #print "min for phase sweep is ",sum_min,' at offset ',min_index
    return min_phase

def est_tone_phase_and_remove(sdata,a,f,sr):
    samples = len(sdata)
    xdata = numpy.zeros(samples)
    min_phase = est_tone_phase(sdata,a,f,sr)
    for i in xrange(samples): xdata[i] = sdata[i] - a*cos(2*pi*i*f/(sr/2.0) + min_phase)
    return (xdata)


def tone_est_near_index(sdata,index,range,sr):
    samples = len(sdata)
    fft_size = 2**int(floor(log(samples)/log(2.0)))
    freq = fft(sdata[0:fft_size])
    pdata = numpy.zeros(fft_size)
    for i in xrange(fft_size): pdata[i] = abs(freq[i])
    peak = 0
    peak_index = 0
    for i in xrange(2*range):
        if (pdata[index+i-range] > peak):
            peak = pdata[index+i-range]
            peak_index = index+i-range

    print "peak is at ",peak_index," and is ",peak        

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

    sum_phase = freq[peak_index-1]*c(-1,d) +    freq[peak_index]*c(0,d) + freq[peak_index+1]*c(1,d)

    sum_c_sq = abs(c(-1,d))*abs(c(-1,d)) +    abs(c(0,d))*abs(c(0,d)) + abs(c(1,d))*abs(c(1,d))

    amp = (abs(sum_phase)/sum_c_sq)/fft_size

    phase_r = cmath.phase(sum_phase)
    freq_est = 0.5*u*sr/fft_size

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

    print "peak is at ",peak_index," and is ",peak        

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

    sum_phase = freq[peak_index-1]*c(-1,d) +    freq[peak_index]*c(0,d) + freq[peak_index+1]*c(1,d)

    sum_c_sq = abs(c(-1,d))*abs(c(-1,d)) +    abs(c(0,d))*abs(c(0,d)) + abs(c(1,d))*abs(c(1,d))

    amp = (abs(sum_phase)/sum_c_sq)/fft_size

    phase_r = cmath.phase(sum_phase)
    freq_est = 0.5*u*sr/fft_size

    return (amp,freq_est,phase_r)

# REMOVAL CASES

def tone_est_and_remove(sdata,sr):
    (a,f,p) = tone_est(sdata,sr)
    xdata = est_tone_phase_and_remove(sdata,a,f,sr)
    print "removed sin with amplitude = ",a, " at frequency ",2*f
    return (xdata,f)

def tone_est_above_index_and_remove(sdata,index,sr):
    (a,f,p) = tone_est_above_index(sdata,index,sr)
    min_phase = est_tone_phase(sdata,a,f,sr)
    samples = len(sdata)
    xdata = numpy.zeros(samples)
    for i in xrange(samples): xdata[i] = sdata[i] - a*cos(2*pi*i*f/(sr/2.0) + min_phase)
    print "removed sin with amplitude = ",a, " at frequency ",2*f
    return (xdata,f)

def tone_est_near_index_and_remove(sdata,index,range,sr):
    (a,f,p) = tone_est_near_index(sdata,index,range,sr)
    min_phase = est_tone_phase(sdata,a,f,sr)
    samples = len(sdata)
    xdata = numpy.zeros(samples)
    for i in xrange(samples): xdata[i] = sdata[i] - a*cos(2*pi*i*f/(sr/2.0) + min_phase)
    print "removed sin with amplitude = ",a, " at frequency ",2*f
    return (xdata,f)

