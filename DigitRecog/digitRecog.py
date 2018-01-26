import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split

# Part 0 - Preprocessing of data
# Importing the dataset

train_set = pd.read_csv('train.csv')
test_set  = pd.read_csv('test.csv')

X_train = (train_set.ix[:,1:].values).astype('float32')
y_train = train_set.ix[:,0].values.astype('int32')
X_test = test_set.values.astype('float32')

#Visualization
#X_train = X_train.reshape(X_train.shape[0], 28, 28)
#for i in range(6, 9):
#    plt.subplot(330 + (i+1))
#    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
#    plt.title(y_train[i]);
    
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape

#Feature standadization
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px

#One hot encoder
from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes

# Part 1 - Building the CNN
# Importing the Keras libraries and packages

model= Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

print("input shape ",model.input_shape)
print("output shape ",model.output_shape)

# Compiling the CNN

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.preprocessing import image
gen = image.ImageDataGenerator()

X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size=0.20,
                                                  random_state=0)
batches = gen.flow(X_train, y_train, batch_size=34)
val_batches=gen.flow(X_val, y_val, batch_size=34)

history=model.fit_generator(batches,
                            batches.n, 
                            nb_epoch=3, 
                            validation_data = val_batches,
                            nb_val_samples = val_batches.n)

predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)
