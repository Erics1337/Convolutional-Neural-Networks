import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split



def CNN_model(num_classes):
    model = Sequential() #part of boilerplate.  Feature selection part here v
    model.add(Conv2D(filters=64, #modulated parameter. generates more features
                kernel_size=(5,5), #filter size. will convol 5 pixels at a time
                border_mode='valid', #whether or not to include border pixels in convolution
                input_shape=(1,32,32), #first layer must specify input dimensions
                data_format='channels_first', #specify # channels comes first in shape
                activation='relu')) #set activation function
    model.add(MaxPooling2D(pool_size=(2,2))) #add 2D Max Pooling layer with 2x2 patch size
    model.add(Dropout(rate=0.2)) #drop 20% of network. Helps reduce overfitting
    model.add(Flatten()) #converts 2D output from previous layer to 1D input for next layer
    #MLP part
    model.add(Dense(units=256, activation='relu')) #add fully-connected layer with 128 neurons
    model.add(Dense(units=num_classes, activation='softmax')) #output layer with 10 neurons
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) #usually vanilla boilerplate
    return model
    

#load training data
print("reading rps-train.csv...")
df = pd.read_csv("rps-train.csv")
X = df.drop(columns="1.00000")
X = X.to_numpy()
X = X.reshape((1749, 1, 32, 32))    #1750, 1224, 1024, 1790976, 
y = df["1.00000"]
y = y.to_numpy()
y = np.reshape(y, (1749,1))

print("Splitting training data using k-fold cross validation...")
X, Xt, y, yt = train_test_split(X, y, test_size=0.3)

print("reading rps-test.csv...")
dfTest = pd.read_csv("rps-test.csv")
Xtest = dfTest
Xtest = Xtest.to_numpy()
Xtest = Xtest.reshape(437, 1, 32, 32)


y = np_utils.to_categorical(y)  #convert to one-hot encoding
num_classes = y.shape[1]    #get number of classes (10)

#create the MLP model and fit
cnn = CNN_model(num_classes)
cnn.fit(X,y, epochs=15)

#make predictions on all test data
yt = cnn.predict(Xt)

#have some fun
#grab a random test instance and answer
import random
i = random.randint(0, Xt.shape[0])
xt = Xt[i]
yp = yt[i]
label = np.argmax(yp)

#visualize
import matplotlib.pyplot as plt
xt = xt.reshape((32,32))
plt.imshow(xt)
plt.title(label)
plt.show()

print("making prediction on test data...")
yp = cnn.predict(Xtest)
p = []
for i in range(1,438):
    yp = np.argmax(y[i,:])
    p.append(yp)
predictions = pd.DataFrame(p)

print("writing predictions to Swanson.csv")
pd.DataFrame(predictions).to_csv('Swanson.csv', header=None, index=None)
