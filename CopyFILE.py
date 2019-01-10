import shutil 
import os

print(os.listdir('/Users/apple/Documents/YL/tez/Datasets/Lungs/small/'))
#print(os.listdir(os.getcwd()))
shutil.copy2('/Users/apple/Documents/YL/tez/Datasets/Lungs/small/177_1b2_Ar_mc_AKGC417L.wav', '/Users/apple/Documents/YL/tez/Datasets/Lungs/small/crackles/177_1b2_Ar_mc_AKGC417L.wav')