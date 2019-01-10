#Label the classes
#first is crackles second is wheezing

'''@TODO: after reading, make a file that lists 
 patient no, chest location, crackles (0/1) wheezing(0/1)
 
 Maybe compare same chest locations amongs themselves for better results?
 
 Maybe eliminate if more than 30 persent is dıfferent?
 
 For now, only one 1 is enought to call it sick.
'''


import errno
import glob
import shutil 
import librosa 
import numpy as np
#import soundfile as sf

path = '/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_final_database/*.txt'
wawpath = '/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_final_database*.wav'
files = glob.glob(path)
x = []
first_result=[]
second_result=[]

def defineClassLabel(first_result,second_result,name):  
    #print('Insıde method')
    #print('first_result',first_result)
    #print('second_result',second_result)
    print('name',name)
    src = '/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_final_database/'
    dstCrackles = src+'crackles/'
    dstWheez = src+'wheezing/'
    dstNone = src+'none/'
    
    sum = 0
    sum2 = 0
    fname = name.split('/')[-1]
    wavname = fname.replace('txt','wav')
    print('fname',fname)
    print('wavname',wavname)
    size = len(first_result)
    
    
    
    #decide = size*30/100
    #If at least one abnormaly is heard, it's enough to identify as sick
    for i in range(size):   
        sum = sum + int(first_result[i])
        #print('fname, sum: ', fname, sum)
        
    if sum >= 1:
        print('\n dstCrackles+fname: ',dstCrackles+fname)
        print('\n dstCrackles+wavname:', dstCrackles+wavname)
        
        #with open(name, 'r') as src, open(dstCrackles+wavname, 'w') as dst: dst.write(src.read())
        shutil.copy2(src+fname, dstCrackles+fname)
        shutil.copy2(src+wavname, dstCrackles+wavname)
        print('Copied file to crackles folder', fname, dstCrackles+fname)
    for i in range(size):   
        sum2 = sum2 + int(second_result[i])
    if sum2 >=1 :
        print('dstWheez+fname: ',dstWheez+fname)
        #with open(name, 'r') as src, open(dstWheez+wavname, 'w') as dst: dst.write(src.read())
        shutil.copy2(src+fname, dstWheez+fname)
        shutil.copy2(src+wavname, dstWheez+wavname)
        print('Copied file to wheezing folder', fname, dstWheez+fname)   
    if sum < 1 and sum2 < 1:
        print('dstNone+fname: ',dstNone+fname)
        #with open(name, 'r') as src, open(dstNone+wavname, 'w') as dst: dst.write(src.read())
        shutil.copy2(src+fname, dstNone+fname)
        shutil.copy2(src+wavname, dstNone+wavname)
        print('Copied file to none folder', fname, dstNone+fname)   
        
        
for name in files:
    try:
        with open(name) as f:
            pass 
            #print(name)
            #for l in f:
            content= f.readlines()
           
            print('for file : ',f , 'the content : ', content)
            
            for i in range(len(content)): 
                data = content[i].split()
                first_result.append(data[2])
                second_result.append(data[3])
              
            defineClassLabel(first_result,second_result,name)    
          
            del first_result[:]
            del second_result[:]
            
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
            
'''
def main():
   # shutil.copy('/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_small_copy/107_3p2_Tc_mc_AKGC417L.wav','/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_final_database/ege' )
    pass


if __name__ == "__main__":
    main()

'''