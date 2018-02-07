import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

print('Inizio programma')

filepath='architecture&synaptic_weights.h5'
train_data_dir = 'data'

##img_path='data/Boots/'+str(1)+'.jpg'
##img = image.load_img(img_path, target_size=(100, 100))
##x = image.img_to_array(img)
##x = np.expand_dims(x, axis=0)
##x_train=np.array(x)
##y_train=np.array([1,0,0,0])
##for i in range(2,11):
##    img_path='data/Boots/'+str(i)+'.jpg'
##    img = image.load_img(img_path, target_size=(100, 100))
##    x = image.img_to_array(img)
##    x = np.expand_dims(x, axis=0)
##    np.append(x_train, x, axis=0)
##    np.append(y_train, [1,0,0,0], axis=0)
##
##for i in range(1,11):
##    img_path='data/Sandals/'+str(i)+'.jpg'
##    img = image.load_img(img_path, target_size=(100, 100))
##    x = image.img_to_array(img)
##    x = np.expand_dims(x, axis=0)
##    np.append(x_train, x, axis=0)
##    np.append(y_train, [0,1,0,0], axis=0)
##
##
##for i in range(1,11):
##    img_path='data/Shoes/'+str(i)+'.jpg'
##    img = image.load_img(img_path, target_size=(100, 100))
##    x = image.img_to_array(img)
##    x = np.expand_dims(x, axis=0)
##    np.append(x_train, x, axis=0)
##    np.append(y_train, [0,0,1,0], axis=0)
##
##
##for i in range(1,11):
##    img_path='data/Slippers/'+str(i)+'.jpg'
##    img = image.load_img(img_path, target_size=(100, 100))
##    x = image.img_to_array(img)
##    x = np.expand_dims(x, axis=0)
##    np.append(x_train, x, axis=0)
##    np.append(y_train, [0,0,0,1], axis=0)
##
##
##print(x_train.shape)

model = Sequential()

#prima convoluzione: input immagini 150x150 e filtro 3x3
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
print(model.input_shape)
print(model.output_shape)

#seconda convolusione: input immagini 148x148 filtro 3x3
model.add(Conv2D(32, (3, 3), activation='relu'))
print(model.output_shape)

#primo pooling di size 2 (immagini si dimezzano quindi 148/2 = 74)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))

#terza convoluzione: input immagini 74x74 filtro 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))
print(model.output_shape)

#quarta convoluzione: input immagini 72x72 con filtro 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))
print(model.output_shape)

#secondo pooling di size 2 72/2=36
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))

model.add(Flatten())
print(model.output_shape)
model.add(Dense(256, activation='relu'))
print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(4, activation='sigmoid'))
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

model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=150)

model.save(filepath)
