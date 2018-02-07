from keras.preprocessing import image
from keras.models import load_model
import numpy as np
filepath='architecture&synaptic_weights.h5'
print("I'm loading the model...")
model = load_model(filepath)

##img_path =input('Insert image name:')
##
##img = image.load_img(img_path, target_size=(150, 150))
##x = image.img_to_array(img)
##x = np.expand_dims(x, axis=0)


for i in range(1,11):
    img_path='data/Boots/'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(preds)
for i in range(1,11):
    img_path='data/Sandals/'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(preds)
for i in range(1,11):
    img_path='data/Shoes/'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(preds)
for i in range(1,11):
    img_path='data/Slippers/'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(preds)
    
#print('I think it is a',decode_predictions(preds, top=1)[0][0][1],'with a',round(100*decode_predictions(preds, top=1)[0][0][2]),'% chance')

print('-----------------------')

