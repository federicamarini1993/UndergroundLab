import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

filepath='architecture&synaptic_weights.h5'
train_data_dir = 'data'

ep=input('Insert Epochs:')
st=input('Insert Step Epochs:')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
print(model.input_shape)
print(model.output_shape)
model.add(Conv2D(32, (3, 3), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
print(model.output_shape)
model.add(Conv2D(64, (3, 3), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))
##Aggiunto il blocco successivo in data 21/02/'18
model.add(Conv2D(128, (3, 3), activation='relu'))
print(model.output_shape)
model.add(Conv2D(128, (3, 3), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)

##Aggiunto il blocco successivo in data 22/02/'18
model.add(Conv2D(256, (3, 3), activation='relu'))
print(model.output_shape)
model.add(Conv2D(256, (3, 3), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)

model.add(Flatten())
print(model.output_shape)
model.add(Dense(256, activation='relu'))
print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))
print(model.output_shape)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

history = LossHistory()

model.fit_generator(
        train_generator,
        steps_per_epoch=int(st),
        epochs=int(ep),
        callbacks=[history])

model.save(filepath)

print(history.losses)
