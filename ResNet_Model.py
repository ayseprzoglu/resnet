
#Datasetoluşturulur
#elimizdeki veriseti split fonksiyonu ile 0.8 train 0.2 test/validation olarak ayırıldı

!pip install split_folders
import splitfolders

input="C:\Users\Casper-X400\Desktop\input"
output="C:\Users\Casper-X400\Desktop\output"
splitfolders.ratio(input, output, seed=112, ratio(.8, .2))

import os
valid_data_dir = r'C:\Users\Casper-X400\Desktop\output\test'
b = os.listdir(valid_data_dir)
print(b)

import os
train_data_dir = r'C:\Users\Casper-X400\Desktop\output\train'
c = os.listdir(train_data_dir)
print(c)

import os
test_data_dir = r'C:\Users\Casper-X400\Desktop\output\test'
d = os.listdir(test_data_dir)
print(d)

from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten

img_height, img_width = (256,256)
batch_size = 16

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    rotation_range=40)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical') #set as training data

valid_generator = train_datagen.flow_from_directory(
    valid_data_dir , #same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode= 'categorical') #set as validation data

test_generator = train_datagen.flow_from_directory(
    test_data_dir,   #same directory as training data
    target_size=(img_height, img_width),
    batch_size=1,                           #test versetinin batch size değeri hep bir olur.
    class_mode= 'categorical') # set as validation

x,y = test_generator.next()
x.shape

base_model = ResNet50(include_top=False, weights= 'imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
metric = tf.keras.metrics.CategoricalAccuracy()

#checkpoiny

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics = metric )

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
        verbose=1, mode='auto', restore_best_weights=True)

history = model.fit(train_generator,
          #validation_data = valid_generator,
          shuffle=True,
          epochs = 50,
          callbacks=[monitor],       
          verbose = 1)

#validation dataseti, test verisetinin öğrenme sırasındaki temsili halidir. Optimizasyon sırasında Kullanılmaz. Background güncellenmesinde kullanılır.

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print('\nTest accuracy:', test_acc)

os.chdir(r"C:\Users\Casper-X400\Desktop\savemodel")
model.save('resnet50')

#Grafiklere dökelim,Gözlemleyelim  ------------------------------------------
import matplotlib.pyplot as plt
acc = history.history['categorical_accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.style.use('seaborn-whitegrid')
plt.figure(dpi = 100)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.legend(loc=0)
plt.savefig(fname="grafiacc.jpg")
plt.show();

------------------------------------------------------------------------------

plt.style.use('seaborn-whitegrid')
plt.figure(dpi = 100)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training  loss')
plt.legend(loc=0)
plt.savefig(fname="grafiloss.jpg")
plt.show();
------------------------------------------------------------------------------

import pandas as pd
import seaborn as sn
import tensorflow as tf

model = tf.keras.models.load_model('resnet50')
filenames = test_generator.filenames
nb_samples = len(test_generator)
y_prob=[]
y_act=[]
test_generator.reset()
for _ in range(nb_samples):
  X_test,Y_test = test_generator.next()
  y_prob.append(model.predict(X_test))
  y_act.append(Y_test)

predicted_class = [list (train_generator.class_indices.keys())[i.argmax()] for i in y_prob]
actual_class = [list (train_generator.class_indices.keys())[i.argmax()] for i in y_act]

out_df = pd.DataFrame(np.vstack([predicted_class,actual_class]).T,columns=['predicted_class','actual_class'])
confusion_matrix = pd.crosstab(out_df['actual_class'],out_df['predicted_class'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix,cmap='Reds', annot=True,fmt='d')
plt.savefig(fname="grafik")
plt.show()
print('test accuracy : {}'.format((np.diagonal(confusion_matrix).sum()/confusion_matrix.sum().sum()*100)))







