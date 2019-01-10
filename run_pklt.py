#!/Users/tkirke/anaconda/bin/python
import time
import sys
import os
from pypklt import *
from lame  import *
from tone_est import *
import numpy
from rescale_audio import rescale_audio_ints,rescale_audio_floats

def check_onset(audio):
        # Determine if there is an 'early' onset
        b_lpf, a_lpf = signal.butter(1, 200/44100.0, 'low')
        sim = numpy.square(audio)
        audio_lpf   = signal.filtfilt(b_lpf,a_lpf,sim)
        thres = 0.015

        invert = 0
        found = 0
        for i in xrange(len(audio)):
                if ((audio_lpf[i] > thres) and not found):
                        if (i < 4000):
                                print "Detected onset for ",fil," at time ",i/44.1," msecs "
                                found = 1
                                invert = 1
        return invert


if __name__ == "__main__":
        mp = re.compile('\.mp3')
        out = re.compile('\.out')
        fil = sys.argv[1]
        if mp.search(fil):
                (root,ext) = fil.split('.')
                sr = 0
                try:
                        (sr,unscaled_audio) = get_samples(fil)
                except:
                        print "problem with "+fil
                if (sr):
                        # first rescale
                        audio = rescale_audio_floats(unscaled_audio)

                        samples = len(audio)
                        fft_size = 2**int(floor(log(samples)/log(2.0)))
                        (peak1,peak1_index,peak2,peak2_index) = find_top_two_peaks(audio)
        
                        peak_diff = 20.0*log(peak1/peak2)/log(10.0)

                        if (peak_diff > 6) and (abs(peak1_index-peak2_index) > 4):
                                print "Peak diff = ",peak_diff, " dB",
                                print " Freq diff = ",sr*(peak1_index-peak2_index)/fft_size
                                (audio,f) = tone_est_and_remove(audio,sr)

                        invert = check_onset(audio)
                        P = Pklt(invert)

                        # then convert from floating point to ints
                        vals = numpy.array(numpy.multiply(2**15,audio),dtype=numpy.int16)
                        res = P.run(vals)
                                
                        (root,ext) = fil.split('.')
                        cleaner = remove_n_harmonics(res,sr,2,50.0)
                        scaled  = rescale_audio_ints(cleaner)
                        save_samples(root+'.out',sr,1,128,scaled)

