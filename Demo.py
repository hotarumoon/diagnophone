'''
Does duration not being the same matters?

Weka?
'''

import matplotlib.pyplot as plt
#import matplotlib.style as ms
#ms.use('seaborn-muted')
import IPython.display as ipd
import sklearn
import librosa 
import librosa.display
import numpy as np
import pandas
import scipy


filename = '107_3p2_Ar_mc_AKGC417L.wav'
secondFilename = '107_3p2_Tc_mc_AKGC417L.wav'

#Load first 3 seconds of audio
x, sr = librosa.load(filename, duration=3)
second_x, second_sr = librosa.load(secondFilename, duration=3)

#Play the audio files:
ipd.Audio(x, rate=sr)
ipd.Audio(x, rate=sr)


#Plot the time-domain waveform of the audio signals:
S_first = librosa.feature.melspectrogram(x, sr=sr)
Sdb_first = librosa.amplitude_to_db(S_first)
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Spectrogram of first sound', fontsize=20)
librosa.display.specshow(Sdb_first, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar()


S_second = librosa.feature.melspectrogram(second_x, sr=second_sr)
Sdb_second = librosa.amplitude_to_db(S_second)
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Spectrogram of second sound', fontsize=20)
librosa.display.specshow(Sdb_second, sr=second_sr, x_axis='time', y_axis='mel')
plt.colorbar()




#Extract Features
#For each segment, compute the MFCCs. 

#Transpose the result to accommodate scikit-learn which assumes that 
#each row is one observation, and each column is one feature dimension


n_mfcc = 12

#Scale the features to have zero mean and unit variance
#Verify that the scaling worked
#Scale the resulting MFCC features to have approximately zero mean and unit 
#variance. Re-use the scaler from above.

#transpose the resulting matrix so that each row is one observation, 
#i.e. one set of MFCCs. Note that the shape and size of the resulting MFCC
# matrix is equivalent to that for the first audio file

scaler = sklearn.preprocessing.StandardScaler()   
  
mfcc_first = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc).T  

mfcc_first_scaled = scaler.fit_transform(mfcc_first)



#Extract MFCCs from the second audio file    
mfcc_second = librosa.feature.mfcc(second_x, sr=second_sr, n_mfcc=n_mfcc).T  


print(mfcc_second.shape)
print(mfcc_first.shape)

#Scale the resulting MFCC features to have approximately zero mean and unit 
#variance. Re-use the scaler from above.
mfcc_second_scaled = scaler.transform(mfcc_second)
print('Second mean ',mfcc_second_scaled.mean(axis=0))
print('Second STD', mfcc_second_scaled.std(axis=0))


labels = np.concatenate((np.zeros(len(mfcc_first_scaled)), np.ones(len(mfcc_second_scaled))))

features = np.vstack((mfcc_first_scaled, mfcc_second_scaled))

#Construct a vector of ground-truth labels, where 0 refers to the crackles 
#audio file, 1 refers to the wheezing audio file and 2 refers to the normal


#Create a classifer model object
# Support Vector Machine
model = sklearn.svm.SVC()

#Train the classifier
model.fit(features, labels)


#Run the Classifier
#To test the classifier, extract an unused 2-second segment from the 
#earlier audio fields as test excerpts
x_first_test, sr_first = librosa.load(filename, duration=2, offset=0)
x_second_test, sr_second = librosa.load(secondFilename, duration=2, offset=0)

#Compute MFCCs from both of the test audio excerpts
mfcc_first_test = librosa.feature.mfcc(x_first_test, sr=sr_first, n_mfcc=n_mfcc).T
mfcc_second_test = librosa.feature.mfcc(x_second_test, sr=sr_second, n_mfcc=n_mfcc).T

print('Shape of first test ', mfcc_first_test.shape)
print('Shape of second test ',mfcc_second_test.shape)

#Scale the MFCCs using the previous scaler 
mfcc_first_test_scaled = scaler.transform(mfcc_first_test)
mfcc_second_test_scaled = scaler.transform(mfcc_second_test)

#Concatenate all test features together
features_test = np.vstack((mfcc_first_test_scaled, mfcc_second_test_scaled))

#Concatenate all test labels together
labels_test = np.concatenate((np.zeros(len(mfcc_first_test)), np.ones(len(mfcc_second_test))))


#Compute the predicted labels
predicted_labels = model.predict(features_test)

#Compute the accuracy score of the classifier on the test data
score = model.score(features_test, labels_test)
print('score: ',score)

#Currently, the classifier returns one prediction for every MFCC vector 
#in the test audio signal. Modify the procedure above such that the classifier 
#returns a single prediction for a 2-second excerpt.
predicted_labels = model.predict(mfcc_first_test_scaled)
r1=np.argmax([(predicted_labels == c).sum() for c in (0, 1)])
print(r1)

predicted_labels = model.predict(mfcc_second_test_scaled)
r2=np.argmax([(predicted_labels == c).sum() for c in (0, 1)])
print(r2)


#Analysis in Pandas
#Read the MFCC features from the first test audio excerpt into a data frame
df_first = pandas.DataFrame(mfcc_first_test_scaled)
print(df_first.shape)
print(df_first.head())


df_second = pandas.DataFrame(mfcc_second_test_scaled)
print(df_second.shape)
print(df_second.head())
#Compute the pairwise correlation of every pair of 12 MFCCs against one another 
#for both test audio excerpts. For each audio excerpt, which pair of MFCCs 
#are the most correlated? least correlated?

print('correlation of first', df_first.corr())
print('correlation of second', df_second.corr())

#Display a scatter plot of any two of the MFCC dimensions
#(i.e. columns of the data frame) against one another. 
#Try for multiple pairs of MFCC dimensions.


df_first.plot.scatter(1, 2, figsize=(7, 7))
df_second.plot.scatter(1, 2, figsize=(7, 7))



#Plot a histogram of all values across a single MFCC, 
#i.e. MFCC coefficient number. Repeat for a few different MFCC numbers
df_first[0].plot.hist(bins=20, figsize=(14, 5))
df_second[0].plot.hist(bins=20, figsize=(14, 5))


df_first[1].plot.hist(bins=20, figsize=(14, 5))
df_second[1].plot.hist(bins=20, figsize=(14, 5))
