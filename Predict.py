from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
filepath='architecture&synaptic_weights.h5'
print("I'm loading the model...")
model = load_model(filepath)

#img_path =input('Insert image name:')

def conv(n):
    if(n==1):
        return 'Boots'
    else:
        if(n==2):
            return 'Sandals'
        else:
            return 'Shoes' 

for i in range(1,31):
    img_path='Test/Real/'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
#    print(i)
#    print(preds)
    print(conv(np.argmax(preds)+1))
    if(i%10==0):
        print('*')
#    print('*')

print('----------')

for i in range(1,11):
    img_path='Test/Boots/'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(conv(np.argmax(preds)+1))
print('----------')
for i in range(1,11):
    img_path='Test/Sandals/'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(conv(np.argmax(preds)+1))
print('----------')
for i in range(1,11):
    img_path='Test/Shoes/'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(conv(np.argmax(preds)+1))
print('----------')
##for i in range(1,11):
##    img_path='Test/Slippers/'+str(i)+'.jpg'
##    img = image.load_img(img_path, target_size=(150, 150))
##    x = image.img_to_array(img)
##    x = np.expand_dims(x, axis=0)
##    preds = model.predict(x)
##    print(np.argmax(preds)+1)
##print('----------')
#print('I think it is a',decode_predictions(preds, top=1)[0][0][1],'with a',round(100*decode_predictions(preds, top=1)[0][0][2]),'% chance')

print('-----------------------')

