'''
DURATION:
librosa.get_duration(y=y, sr=sr)

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

import tflearn
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn import tree





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
files_crackles = librosa.util.find_files(crackles)
files_wheezing = librosa.util.find_files(wheezing)
files_none = librosa.util.find_files(none)

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

w, wsr = librosa.load('/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_final_database/wheezing/161_1b1_Pl_sc_LittC2SE.wav',duration=5 )
c, csr = librosa.load('/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_final_database/crackles/175_1b1_Lr_sc_Litt3200.wav',duration=5 )
n, nsr = librosa.load('/Users/apple/Documents/YL/tez/Datasets/Lungs/ICBHI_final_database/none/204_7p5_Lr_mc_AKGC417L.wav',duration=5 )



SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#Plot the time-domain waveform of the audio signals:
fig = plt.figure(figsize=(14, 5))
ax2 = plt.axes()
ax2.set_xlabel('Time', fontsize=24)
ax2.set_ylabel('Amplitude', fontsize=24)
librosa.display.waveplot(c, csr)

fig = plt.figure(figsize=(14, 5))
ax2 = plt.axes()
ax2.set_xlabel('Time', fontsize=24)
ax2.set_ylabel('Amplitude', fontsize=24)
librosa.display.waveplot(w, wsr)

fig = plt.figure(figsize=(14, 5))
ax2 = plt.axes()
ax2.set_xlabel('Time', fontsize=24)
ax2.set_ylabel('Amplitude', fontsize=24)
librosa.display.waveplot(n, nsr)

#Plot the spectrograms:
S_crackles = librosa.feature.melspectrogram(c, sr=csr)
Sdb_crackles = librosa.amplitude_to_db(S_crackles)
fig = plt.figure(figsize=(15, 5))
ax2 = plt.axes()
ax2.set_xlabel('Time', fontsize=24)
ax2.set_ylabel('Frequency', fontsize=24)
librosa.display.specshow(Sdb_crackles, sr=csr, x_axis='time', y_axis='mel')
plt.colorbar()

S_wheezing = librosa.feature.melspectrogram(w, sr=wsr)
Sdb_wheezing = librosa.amplitude_to_db(S_wheezing)
fig = plt.figure(figsize=(15, 5))
ax2 = plt.axes()
ax2.set_xlabel('Time', fontsize=24)
ax2.set_ylabel('Frequency', fontsize=24)
librosa.display.specshow(Sdb_wheezing, sr=wsr, x_axis='time', y_axis='mel')
plt.colorbar()

S_none = librosa.feature.melspectrogram(n, sr=nsr)
Sdb_none = librosa.amplitude_to_db(S_none)
fig = plt.figure(figsize=(15, 5))
ax2 = plt.axes()
ax2.set_xlabel('Time', fontsize=24)
ax2.set_ylabel('Frequency', fontsize=24)
librosa.display.specshow(Sdb_none, sr=nsr, x_axis='time', y_axis='mel')
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
kf = KFold(n_splits=5)
kf.get_n_splits(features)

print(kf)  
KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


#Create a classifer model object
# Support Vector Machine
#model = sklearn.svm.SVC()

#model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64
width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

### add this "fix" for tensorflow version errors
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x ) 


model = tflearn.DNN(net, tensorboard_verbose=0)
while 1: #training_iters
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
  _y=model.predict(X)



#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#model = tree.DecisionTreeClassifier()

#Train the classifier
#model.fit(X_train, y_train)
model.fit(features, labels)

#Run the Classifier
#Compute the predicted labels
predicted_labels = model.predict(X_test)

#Compute the accuracy score of the classifier on the test data
score = model.score(X_test, y_test)
print('score: ',score)

model.save('SVM model')


#CONFUSION MATRIX!
print('confusion matrix')
print(confusion_matrix(y_test, predicted_labels))

'''

#Currently, the classifier returns one prediction for every MFCC vector 
#in the test audio signal. Modify the procedure above such that the classifier 
#returns a single prediction for test excerpt.
arrayyed=np.array(mfcc_crackles_scaled)
predicted_labels_c = model.predict(np.array(mfcc_crackles_scaled))
r1=np.argmax([(predicted_labels_c == c).sum() for c in (0, 1)])
print(r1)

predicted_labels_w = model.predict(np.array(mfcc_wheezing_scaled))
r2=np.argmax([(predicted_labels_w == c).sum() for c in (0, 1)])
print(r2)


predicted_labels_n = model.predict(np.array(mfcc_none_scaled))
r2=np.argmax([(predicted_labels_n == c).sum() for c in (0, 1)])
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

print('df_crackles: ')
df_crackles.plot.scatter(1, 2, figsize=(7, 7))
print('df_crackles: ')
df_crackles.plot.scatter(2, 3, figsize=(7, 7))
print('df_wheezing: ')
df_wheezing.plot.scatter(1, 2, figsize=(7, 7))
print('df_none: ')
df_none.plot.scatter(1, 2, figsize=(7, 7))



#Plot a histogram of all values across a single MFCC, 
#i.e. MFCC coefficient number. Repeat for a few different MFCC numbers
df_crackles[0].plot.hist(bins=20, figsize=(14, 5))
df_wheezing[0].plot.hist(bins=20, figsize=(14, 5))
df_none[0].plot.hist(bins=20, figsize=(14, 5))


df_crackles[1].plot.hist(bins=20, figsize=(14, 5))
df_wheezing[1].plot.hist(bins=20, figsize=(14, 5))
df_none[1].plot.hist(bins=20, figsize=(14, 5))

