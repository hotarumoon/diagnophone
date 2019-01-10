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
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

crackles = '/Users/apple/Documents/YL/tez/Datasets/Lungs/small/crackles'
wheezing = '/Users/apple/Documents/YL/tez/Datasets/Lungs/small/wheezing'
none = '/Users/apple/Documents/YL/tez/Datasets/Lungs/small/none'


x_crackles=[]
sr_crackles=[]

x_wheezing=[]
sr_wheezing=[]

x_none=[]
sr_none=[]

#Load audio
files_crackles = librosa.util.find_files(crackles,ext='wav')
files_wheezing = librosa.util.find_files(wheezing,ext='wav')
files_none = librosa.util.find_files(none,ext='wav')

print(len(files_wheezing))
print(len(files_crackles))
print(len(files_none))

#Read all wheezing files
for i in range(len(files_wheezing)):
    fname = files_wheezing[i].split('/')[-1]
    #print('Reading file ',files_wheezing[i])
    x, sr = librosa.load(files_wheezing[i],duration=10 )
    #x, sr = scipy.io.wavfile.read(files_crackles[i])
    x_wheezing.append(x)
    sr_wheezing.append(sr)
   
#Read all crackles files
for i in range(len(files_crackles)):
    fname = files_crackles[i].split('/')[-1]
    x, sr = librosa.load(files_crackles[i],duration=10 )
    x_crackles.append(x)
    sr_crackles.append(sr)

#Read all normal files
for i in range(len(files_none)):
    fname = files_none[i].split('/')[-1]
    x, sr = librosa.load(files_none[i],duration=10 )
    x_none.append(x)
    sr_none.append(sr)


######
'''
#Plot the time-domain waveform of the audio signals:
fig = plt.figure(figsize=(14, 5))
fig.suptitle('Time-domain waveform of a crackles', fontsize=20)
librosa.display.waveplot(x_crackles[0], sr_crackles[0])

fig = plt.figure(figsize=(14, 5))
fig.suptitle('Time-domain waveform of a wheezing sound', fontsize=20)
librosa.display.waveplot(x_wheezing[0], sr_wheezing[0])

fig = plt.figure(figsize=(14, 5))
fig.suptitle('Time-domain waveform of a normal sound', fontsize=20)
librosa.display.waveplot(x_none[0], sr_none[0])

#Plot the spectrograms:
S_crackles = librosa.feature.melspectrogram(x_crackles[0], sr=sr_crackles[2])
Sdb_crackles = librosa.amplitude_to_db(S_crackles)
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Spectrogram of a crackles sound', fontsize=20)
librosa.display.specshow(Sdb_crackles, sr=sr_crackles[2], x_axis='time', y_axis='mel')
plt.colorbar()

S_wheezing = librosa.feature.melspectrogram(x_wheezing[2], sr=sr_wheezing[2])
Sdb_wheezing = librosa.amplitude_to_db(S_wheezing)
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Spectrogram of wheezing sound', fontsize=20)
librosa.display.specshow(Sdb_wheezing, sr=sr_wheezing[2], x_axis='time', y_axis='mel')
plt.colorbar()

S_none = librosa.feature.melspectrogram(x_none[2], sr=sr_none[2])
Sdb_none = librosa.amplitude_to_db(S_none)
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Spectrogram of normal sound', fontsize=20)
librosa.display.specshow(Sdb_none, sr=sr_none[0], x_axis='time', y_axis='mel')
plt.colorbar()

'''


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
 
mfcc_crackles=[]
mfcc_crackles_scaled=[]

for i in range(len(x_crackles)):
    mfcc_crackles.append(librosa.feature.mfcc(x_crackles[i], sr=sr_crackles[i], n_mfcc=n_mfcc).T)
    mfcc_crackles_scaled.append(scaler.fit_transform(mfcc_crackles[i]))
    #print('Mean crackles ',i,':', mfcc_crackles_scaled[i].mean(axis=0))
    #print('STD crackles ',i,':', mfcc_crackles_scaled[i].std(axis=0))

mfcc_wheezing=[]
mfcc_wheezing_scaled=[]
for i in range(len(x_wheezing)):
    mfcc_wheezing.append(librosa.feature.mfcc(x_wheezing[i], sr=sr_wheezing[i], n_mfcc=n_mfcc).T)
    mfcc_wheezing_scaled.append(scaler.fit_transform(mfcc_wheezing[i]))
    #print('Mean wheezing ',i,':', mfcc_wheezing_scaled[i].mean(axis=0))
    #print('STD wheezing ',i,':', mfcc_wheezing_scaled[i].std(axis=0))

mfcc_none=[]
mfcc_none_scaled=[]
for i in range(len(x_none)):
    mfcc_none.append(librosa.feature.mfcc(x_none[i], sr=sr_none[i], n_mfcc=n_mfcc).T)
    mfcc_none_scaled.append(scaler.fit_transform(mfcc_none[i]))
    #print('Mean none ',i,':', mfcc_none_scaled[i].mean(axis=0))
    #print('STD none ',i,':', mfcc_none_scaled[i].std(axis=0))

   
mfcc_features=np.ones_like(mfcc_crackles_scaled[0])
#Train a Classifier
#Concatenate all of the scaled feature vectors into one feature table
for i in range(len(mfcc_crackles_scaled)):
    mfcc_features=np.vstack(np.array(mfcc_crackles_scaled[i]))
    

mfcc_crackles_stacked=np.concatenate(mfcc_crackles_scaled, axis=0)
mfcc_wheezing_stacked=np.concatenate(mfcc_wheezing_scaled, axis=0)
mfcc_none_stacked=np.concatenate(mfcc_none_scaled, axis=0)

#Construct a vector of ground-truth labels, where 0 refers to the crackles 
#audio file, 1 refers to the wheezing audio file and 2 refers to the normal
labels = np.concatenate((np.zeros(len(mfcc_crackles_stacked)), np.ones(len(mfcc_wheezing_stacked)),(np.ones(len(mfcc_none_stacked))*2)))

features = np.concatenate((np.concatenate((mfcc_crackles_stacked,mfcc_wheezing_stacked),axis=0),mfcc_none_stacked),axis=0)



#Kfold Cross validation
kf = KFold(n_splits=3)
kf.get_n_splits(features)

print(kf)  
KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

'''
#Create a classifer model object
# Support Vector Machine
model = sklearn.svm.SVC()

#Train the classifier
#model.fit(X_train, y_train)
model.fit(features, labels)

#Run the Classifier
#Compute the predicted labels
predicted_labels = model.predict(X_test)

#Compute the accuracy score of the classifier on the test data
score = model.score(X_test, y_test)
print('score: ',score)
'''


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(features, labels)                         

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

predicted_labels=clf.predict(X_test)
score = clf.score(X_test, y_test)
print('score: ',score)

#CONFUSION MATRIX!
print('confusion matrix')
print(confusion_matrix(y_test, predicted_labels))

'''
#Currently, the classifier returns one prediction for every MFCC vector 
#in the test audio signal. Modify the procedure above such that the classifier 
#returns a single prediction for test excerpt.
predicted_labels = model.predict(mfcc_first_test_scaled)
r1=np.argmax([(predicted_labels == c).sum() for c in (0, 1)])
print(r1)

predicted_labels = model.predict(mfcc_second_test_scaled)
r2=np.argmax([(predicted_labels == c).sum() for c in (0, 1)])
print(r2)
'''

#Analysis in Pandas
#Read the MFCC features from the first test audio excerpt into a data frame
df_crackles = pandas.DataFrame(mfcc_crackles_scaled[0])
#print(df_crackles.shape)
#print(df_crackles.head())


df_wheezing = pandas.DataFrame(mfcc_wheezing_scaled[0])
#print(df_wheezing.shape)
#print(df_wheezing.head())

df_none = pandas.DataFrame(mfcc_none_scaled[0])
#print(df_none.shape)
#print(df_none.head())


#Compute the pairwise correlation of every pair of 12 MFCCs against one another 
#for both test audio excerpts. For each audio excerpt, which pair of MFCCs 
#are the most correlated? least correlated?
#print('')
#print('correlation of crackles', df_crackles.corr())
#print('')
#print('correlation of wheezing', df_wheezing.corr())
#print('correlation of normal', df_none.corr())
#print('')


#Display a scatter plot of any two of the MFCC dimensions
#(i.e. columns of the data frame) against one another. 
#Try for multiple pairs of MFCC dimensions.


df_crackles.plot.scatter(1, 2, figsize=(7, 7))
df_wheezing.plot.scatter(1, 2, figsize=(7, 7))
df_none.plot.scatter(1, 2, figsize=(7, 7))



#Plot a histogram of all values across a single MFCC, 
#i.e. MFCC coefficient number. Repeat for a few different MFCC numbers
df_crackles[0].plot.hist(bins=20, figsize=(14, 5))
df_wheezing[0].plot.hist(bins=20, figsize=(14, 5))
df_none[0].plot.hist(bins=20, figsize=(14, 5))


df_crackles[1].plot.hist(bins=20, figsize=(14, 5))
df_wheezing[1].plot.hist(bins=20, figsize=(14, 5))
df_none[1].plot.hist(bins=20, figsize=(14, 5))

