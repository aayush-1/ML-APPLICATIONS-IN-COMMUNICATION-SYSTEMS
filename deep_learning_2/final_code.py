#complete
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import cmath
from sklearn import metrics 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout

from sklearn.model_selection import train_test_split

# Read data
data=pd.read_csv('trainDataLabels.csv')
data=data.replace('i','j',regex=True)
testdata=pd.read_csv('testData.csv')
testdata=testdata.replace('i','j',regex=True)


X_train=data.values[:,2:102]
y_train=data.values[:,102]
X_test=testdata.values[:,2:102]


x_train=np.zeros((X_train.shape[0]*100,2))
Y_train=np.zeros((y_train.shape[0]*100))
l=0
for i in range(X_train.shape[0]):
    for j in range(100):
        x_train[l,0]=complex(X_train[i,j]).real
        Y_train[l]=y_train[i]
        l=l+1
            
l=0
for i in range(X_train.shape[0]):
    for j in range(100):
        x_train[l,1]=complex(X_train[i,j]).imag
        l=l+1
        
        
        
x_test=np.zeros((X_test.shape[0]*100,2))
l=0
for i in range(X_test.shape[0]):
    for j in range(100):
        x_test[l,0]=complex(X_test[i,j]).real
        l=l+1
            
l=0
for i in range(X_test.shape[0]):
    for j in range(100):
        x_test[l,1]=complex(X_test[i,j]).imag     
        l=l+1

Y_train=Y_train.reshape(-1,1)

data1=np.hstack((x_train,Y_train))


np.random.shuffle(data1)
            
xtrain,xval,ytrain,yval=train_test_split(data1[:,:2],data1[:,2],test_size=0.25)



ytrain=ytrain-1
labels=3
ytrain=np_utils.to_categorical(ytrain,labels)



from sklearn import metrics
input_dim = xtrain.shape[1]
nb_classes = ytrain.shape[1]



model = Sequential()
model.add(Dense(130, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

print("Training...")
history=model.fit(xtrain, ytrain, epochs=420, batch_size=50, validation_split=0.32 , verbose=1)

print("Generating test predictions...")
preds = model.predict_classes(xval, verbose=1)
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

ypredicted=model.predict_classes(x_test)
ypredicted=ypredicted+1
print(ypredicted)

from collections import defaultdict

ypredi=[]

for i in range(int(ypredicted.shape[0]/100)):
    L = ypredicted[i*100:i*100+99]

    d = defaultdict(int)
    for j in L:
        d[j] += 1
    result = max(d.items(), key=lambda x: x[1])
    ypredi.append(result[0])


ypred=np.asarray(ypredi)
print("ypredicted: ",ypred)
ypred=np.reshape(ypred,(-1,1))
Id=np.linspace(1,len(ypred+1),len(ypred))
Id=np.reshape(Id,(-1,1))
Id=Id.astype(int)
ypred=np.hstack((Id,ypred))
df = pd.DataFrame(ypred,columns = ["Id","Predicted Label"])
df.to_csv("submission2.csv", sep=',', encoding='utf-8', index = False)
