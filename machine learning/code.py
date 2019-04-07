import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn import neighbors
import cmath
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
data=pd.read_csv('trainDataLabels.csv')
data=data.replace('i','j',regex=True)
testdata=pd.read_csv('testData.csv')
testdata=testdata.replace('i','j',regex=True)
xtest=testdata.values[:,1]
xtrain,xval,ytrain,yval=train_test_split(data.values[:,1],data.values[:,2],test_size=0.25,random_state=12)
ytrain=ytrain.astype('int')
yval=yval.astype('int')
xtrain=xtrain.reshape(-1,1)
xval=xval.reshape(-1,1)
xtest=xtest.reshape(-1,1)


I_train=np.zeros((xtrain.shape))
Q_train=np.zeros((xtrain.shape))

for i in range(0,xtrain.shape[0]):
    a=complex(xtrain[i]).real    
    b=complex(xtrain[i]).imag
    I_train[i]=a
    Q_train[i]=b



    
    
I_val=np.zeros((xval.shape))    
Q_val=np.zeros((xval.shape))

for i in range(0,xval.shape[0]):
    a=complex(xval[i]).real
    b=complex(xval[i]).imag
    I_val[i]=a
    Q_train[i]=b

    


    
    
I_test=np.zeros((xtest.shape))    
Q_test=np.zeros((xtest.shape))

for i in range(0,xtest.shape[0]):
    a=complex(xtest[i]).real
    b=complex(xtest[i]).imag
    I_test[i]=a
    Q_test[i]=b

    



training_x=np.hstack((Q_train,I_train))

training_y=ytrain


validation_x=np.hstack((Q_val,I_val))

validation_y=yval



testing_x=np.hstack((Q_test,I_test))




clf = MLPClassifier(learning_rate_init=0.01,solver='adam',max_iter=1000,hidden_layer_sizes=(10,6),learning_rate='adaptive')
clf.fit(training_x,training_y)
ypredicted=clf.predict(validation_x)
print(metrics.accuracy_score(validation_y, ypredicted))

ypredicted=clf.predict(testing_x)
ypredicted=np.reshape(ypredicted,(-1,1))
Id=np.linspace(1,len(ypredicted+1),len(ypredicted))
Id=np.reshape(Id,(-1,1))
Id=Id.astype(int)
ypredicted=np.hstack((Id,ypredicted))
df = pd.DataFrame(ypredicted,columns = ["Id","Predicted Label"])
df.to_csv("submission.csv", sep=',', encoding='utf-8', index = False)





