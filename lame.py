#coding:utf-8
import time
import ctypes
import numpy
from scipy.io import wavfile
import struct
from math import floor

def scale_then_save_samples(inputFile,sample_rate,channel_count,bit_rate,audio_data):
    (root,ext) = inputFile.split('.')
    if (ext == 'wav'):
        if (audio_data.dtype == numpy.int16):
            sar = audio_data
        else:
            x_max = numpy.amax(numpy.absolute(audio_data))
            gain = (2**15)*0.8/x_max
            sar = numpy.array(numpy.multiply(gain,audio_data),dtype=numpy.int16)
            print "Autoscale for .wav format, gain = ",gain
    else:
        sar = audio_data
    save_samples(inputFile,sample_rate,channel_count,bit_rate,sar)


def save_samples(inputFile,sample_rate,channel_count,bit_rate,audio_data):
    (root,ext) = inputFile.split('.')
    if (ext == 'wav'):
        if (audio_data.dtype == numpy.int16):
            # assume already scaled
            wavfile.write(inputFile,sample_rate,audio_data)
        else:
            print "data type is ",audio_data.dtype, " converting to int16 for .wav"
            sar = numpy.array(audio_data,dtype=numpy.int16)
            wavfile.write(inputFile,sample_rate,sar)
    else:
        encode_mp3(inputFile,sample_rate,channel_count,bit_rate,audio_data)

def get_samples(inputFile):
    fields = inputFile.split('.')
    ext = fields[-1]
    if (ext == 'wav'):
        (sr,inp) = wavfile.read(inputFile)
        # scale to -1 to 1 range
        gain = (2**-15)
        inp = numpy.array(numpy.multiply(gain,inp))
        
        return(sr,inp)
    elif (ext == 'mp3'):
        sr = 44100 # fix later TBD
        return (sr,decode_mp3(inputFile))


def encode_mp3(outputFile,sample_rate,channel_count,bit_rate,audio_data):
    lame = LameEncoder(sample_rate,channel_count, bit_rate)
    output_file  = open(outputFile, "wb")
    b = ''
    l = audio_data.size
    for i in xrange(l):
        b += struct.pack("i",audio_data[i])
    output = lame.encode(b,output_file)
    output_file.close()

def decode_mp3(inputFile):
    lame = LameDecoder()
    input_file  = open(inputFile, "rb")
    vals = []
    done = False # this mattered??
    while 1:
        data = input_file.read(512)
        if data:
            output = lame.decode(data)
            if output:
                vals.extend(output)
                while 1:
                    output = lame.flush()
                    if output:
                        vals.extend(output)
                    else:
                        break
        else:
            input_file.close()
            break
    size = len(vals)
    # convert from list to numpy array
    pdata = numpy.zeros(size)
    i = 0
    for v in vals:
        pdata[i] = v*(2**-31.0)
        i = i+1
    return pdata

class LameDecoder():
    def __init__(self):
        self.dll  = ctypes.CDLL("/usr/local/lib/libmp3lame.dylib")
        self.dll.lame_init.restype = ctypes.c_void_p;
        self.dll.lame_set_decode_only.argtypes = [ctypes.c_void_p, ctypes.c_int];
        self.dll.hip_decode1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.dll.hip_decode_init.restype = ctypes.c_void_p;
        self.dll.lame_init_params.argtypes = [ctypes.c_void_p];
        self.dll.lame_get_framesize.argtypes = [ctypes.c_void_p]
        self.lame = self.dll.lame_init()
        self.dll.lame_set_decode_only(self.lame,ctypes.c_int(1))
        self.dll.lame_init_params(self.lame)
        self.hip  = self.dll.hip_decode_init()
        self.dll.lame_close.argtypes = [ctypes.c_void_p]
        self.dll.lame_close.restype = ctypes.c_int

    def decode(self, mp3_data):
        #print "calling decode"
        output_buff_len =  self.dll.lame_get_framesize(self.lame) 
        output_buff     = (ctypes.c_int*output_buff_len)()
        output_size     = self.dll.hip_decode1(self.hip, mp3_data, ctypes.c_int(len(mp3_data)), output_buff, 0)/2
        return output_buff[:output_size]
        
    def flush(self):
        output_buff_len =  self.dll.lame_get_framesize(self.lame) 
        output_buff     = (ctypes.c_int*output_buff_len)() 
        output_size     = self.dll.hip_decode1(self.hip, ctypes.create_string_buffer(""), 0, output_buff, 0)/2
        return output_buff[:output_size]

    def __del__(self):
        self.dll.lame_close(self.lame)
    
    
class LameEncoder():
    def __init__(self, sample_rate, channel_count, bit_rate):
        self.dll  = ctypes.CDLL("/usr/local/lib/libmp3lame.dylib")
        self.dll.lame_init.restype = ctypes.c_void_p;
        self.lame = self.dll.lame_init()
        self.dll.lame_set_in_samplerate.argtypes = [ctypes.c_void_p, ctypes.c_int];
        self.dll.lame_set_in_samplerate(self.lame, sample_rate);
        self.dll.lame_set_num_channels.argtypes = [ctypes.c_void_p, ctypes.c_int];
        self.dll.lame_set_num_channels(self.lame, channel_count);
        self.dll.lame_set_brate.argtypes = [ctypes.c_void_p, ctypes.c_int];
        self.dll.lame_set_brate(self.lame, bit_rate);
        self.dll.lame_set_quality.argtypes = [ctypes.c_void_p, ctypes.c_int];
        self.dll.lame_set_quality(self.lame, 3);
        self.dll.lame_init_params.argtypes = [ctypes.c_void_p];
        self.dll.lame_init_params(self.lame);

    def encode(self, pcm_data, fn):
        sample_count    = len(pcm_data) /2
        output_buff_len = int(1.25 * sample_count + 7200)
        output_buff     = (ctypes.c_char*output_buff_len)()
        self.dll.lame_encode_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int];
        output_size     = self.dll.lame_encode_buffer(self.lame, pcm_data, 0, sample_count, output_buff, output_buff_len);
        if (output_size): fn.write(output_buff[0:output_size])

if __name__ == "__main__":
    import pylab
    print "test decoding sin.mp3 ..."
    vals = decode_mp3("sin.mp3")
    # convert from floating point to ints
    int_vals = numpy.floor(vals*2**31)
    print "now encoding ",len(vals)," samples to sin_x.mp3"
    encode_mp3('sin_x.mp3',44100,1,128,int_vals)
    print "plotting"
    pylab.ion()
    pylab.plot(vals,'r.')
    pylab.grid(True)
    pylab.show()

