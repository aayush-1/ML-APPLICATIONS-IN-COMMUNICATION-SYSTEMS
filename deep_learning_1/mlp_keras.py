#complete
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
import cmath

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split

# Read data
data=pd.read_csv('trainDataLabels.csv')
data=data.replace('i','j',regex=True)
testdata=pd.read_csv('testData.csv')
testdata=testdata.replace('i','j',regex=True)

X_test=testdata.values[:,1]
X_train,xval,ytrain,yval=train_test_split(data.values[:,1],data.values[:,2],test_size=0.25)



ytrain=ytrain.astype('int')
yval=yval.astype('int')
X_train=X_train.reshape(-1,1)
xval=xval.reshape(-1,1)
X_test=X_test.reshape(-1,1)


ytrain=ytrain-1

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(ytrain)





I_train=np.zeros((X_train.shape))
Q_train=np.zeros((X_train.shape))
valuetrain=np.zeros((X_train.shape))
for i in range(0,X_train.shape[0]):
    a=complex(X_train[i]).real    
    b=complex(X_train[i]).imag
    I_train[i]=a
    Q_train[i]=b
    valuetrain[i]=np.sqrt(a**2+b**2)

phasetrain=np.zeros((X_train.shape))
for i in range(0,X_train.shape[0]):
    phasetrain[i]=cmath.phase(X_train[i])
    
    
I_val=np.zeros((xval.shape))    
Q_val=np.zeros((xval.shape))
valueval=np.zeros((xval.shape))
for i in range(0,xval.shape[0]):
    a=complex(xval[i]).real
    b=complex(xval[i]).imag
    I_val[i]=a
    Q_train[i]=b
    valueval[i]=np.sqrt(a**2+b**2)
    
    
phaseval=np.zeros((xval.shape))
for i in range(0,xval.shape[0]):
    phaseval[i]=cmath.phase(xval[i])
    
I_test=np.zeros((X_test.shape))    
Q_test=np.zeros((X_test.shape))
valuetest=np.zeros((X_test.shape))
for i in range(0,X_test.shape[0]):
    a=complex(X_test[i]).real
    b=complex(X_test[i]).imag
    I_test[i]=a
    Q_test[i]=b
    valuetest[i]=np.sqrt(a**2+b**2)
    
    
phasetest=np.zeros((X_test.shape))
for i in range(0,X_test.shape[0]):
    phasetest[i]=cmath.phase(X_test[i])
    
    
X_train=np.hstack((Q_train,I_train,phasetrain,valuetrain))
X_val=np.hstack((Q_val,I_val,phaseval,valueval))
X_test=np.hstack((Q_test,I_test,phasetest,valuetest))
print(X_test.shape)

from sklearn import metrics
input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]


model = Sequential()
model.add(Dense(120, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(53))
model.add(Activation('relu'))
model.add(Dropout(0.12))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

print("Training...")
history=model.fit(X_train, y_train, epochs=400, batch_size=100, validation_split=0.32 , verbose=1)

print("Generating test predictions...")
preds = model.predict_classes(X_val, verbose=1)
preds=preds+1

print(metrics.accuracy_score(yval, preds))


import matplotlib.pyplot as plt


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


print("preds: ",preds)
ypredicted=model.predict_classes(X_test)
ypredicted=ypredicted+1
print("ypredicted: ",ypredicted)
ypredicted=np.reshape(ypredicted,(-1,1))
Id=np.linspace(1,len(ypredicted+1),len(ypredicted))
Id=np.reshape(Id,(-1,1))
Id=Id.astype(int)
ypredicted=np.hstack((Id,ypredicted))
df = pd.DataFrame(ypredicted,columns = ["Id","Predicted Label"])
df.to_csv("submission.csv", sep=',', encoding='utf-8', index = False)
