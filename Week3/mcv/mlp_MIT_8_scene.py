import os
import getpass


from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import svm_function

#user defined variables
IMG_SIZE    = [8,16,24,32,48,64,80,96]
BATCH_SIZE  = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
#MODEL_FNAME = '/home/grupo01/mcv/models/Josep/my_first_mlp.h5'

IMG_SIZES_ACCURACY = []

for size in IMG_SIZE:
  if not os.path.exists(DATASET_DIR):
    print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
    quit()

  MODEL_FNAME = '/home/grupo01/mcv/models/Sergio/size_' + str(size) + '_BATCH_' + str(BATCH_SIZE) + '.h5'
  print('Building MLP model...\n')


  #Build the Multi Layer Perceptron model
  model = Sequential()
  model.add(Reshape((size*size*3,),input_shape=(size, size, 3),name='first'))
  model.add(Dense(units=2048, activation='relu',name='second'))
  #model.add(Dense(units=1024, activation='relu'))
  model.add(Dense(units=8, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])

  print(model.summary())
  plot_model(model, to_file='/home/grupo01/mcv/models/Sergio/modelMLP_size_'+str(size)+'_BATCH_'+str(BATCH_SIZE) + '.png', show_shapes=True, show_layer_names=True)

  print('Done!\n')

  if os.path.exists(MODEL_FNAME):
    print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

  print('Start training...\n')

  # this is the dataset configuration we will use for training
  # only rescaling
  train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)

  # this is the dataset configuration we will use for testing:
  # only rescaling
  test_datagen = ImageDataGenerator(rescale=1./255)

  # this is a generator that will read pictures found in
  # subfolers of 'data/train', and indefinitely generate
  # batches of augmented image data
  train_generator = train_datagen.flow_from_directory(
          DATASET_DIR+'/train',  # this is the target directory
          target_size=(size, size),  # all images will be resized to IMG_SIZExIMG_SIZE
          batch_size=BATCH_SIZE,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

  # this is a similar generator, for validation data
  validation_generator = test_datagen.flow_from_directory(
          DATASET_DIR+'/test',
          target_size=(size, size),
          batch_size=BATCH_SIZE,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')

  history = model.fit_generator(
          train_generator,
          steps_per_epoch=1881 // BATCH_SIZE,
          epochs=50,
          validation_data=validation_generator,
          validation_steps=807 // BATCH_SIZE,
          verbose=0)

  print('Done!\n')
  print('Saving the model into '+MODEL_FNAME+' \n')
  model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
  print('Done!\n')

    # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('/home/grupo01/mcv/models/Sergio/size_' + str(size) + '_BATCH_' + str(BATCH_SIZE) + '_accuracy.jpg')
  plt.close()
    # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('/home/grupo01/mcv/models/Sergio/size_' + str(size) + '_BATCH_' + str(BATCH_SIZE) + '_loss.jpg')
  plt.close()

  IMG_SIZES_ACCURACY.append(history.history['acc'])
  print(IMG_SIZES_ACCURACY[0])


# Now fized image size, changing BATCH_SIZE
BATCH_SIZES = [128]
max_value = max(IMG_SIZES_ACCURACY)
max_index = IMG_SIZES_ACCURACY.index(max_value)
IMAGE_SIZE = IMG_SIZE[max_index]


for batch in BATCH_SIZES:
  if not os.path.exists(DATASET_DIR):
    print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
    quit()

  MODEL_FNAME = '/home/grupo01/mcv/models/Sergio/size_' + str(IMAGE_SIZE) + '_BATCH_' + str(batch) + '.h5'
  print('Building MLP model...\n')


  #Build the Multi Layer Perceptron model
  model = Sequential()
  model.add(Reshape((IMAGE_SIZE*IMAGE_SIZE*3,),input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),name='first'))
  model.add(Dense(units=2048, activation='relu',name='second'))
  #model.add(Dense(units=1024, activation='relu'))
  model.add(Dense(units=8, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])

  print(model.summary())
  plot_model(model, to_file='/home/grupo01/mcv/models/Sergio/modelMLP_size'+str(IMAGE_SIZE)+'_BATCH_'+str(batch)+'.png', show_shapes=True, show_layer_names=True)

  print('Done!\n')

  if os.path.exists(MODEL_FNAME):
    print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

  print('Start training...\n')

  # this is the dataset configuration we will use for training
  # only rescaling
  train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)

  # this is the dataset configuration we will use for testing:
  # only rescaling
  test_datagen = ImageDataGenerator(rescale=1./255)

  # this is a generator that will read pictures found in
  # subfolers of 'data/train', and indefinitely generate
  # batches of augmented image data
  train_generator = train_datagen.flow_from_directory(
          DATASET_DIR+'/train',  # this is the target directory
          target_size=(IMAGE_SIZE, IMAGE_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
          batch_size=batch,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

  # this is a similar generator, for validation data
  validation_generator = test_datagen.flow_from_directory(
          DATASET_DIR+'/test',
          target_size=(IMAGE_SIZE, IMAGE_SIZE),
          batch_size=batch,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')

  history = model.fit_generator(
          train_generator,
          steps_per_epoch=1881 // batch,
          epochs=50,
          validation_data=validation_generator,
          validation_steps=807 // batch,
          verbose=0)

  print(train_generator)
  print('Done!\n')
  print('Saving the model into '+MODEL_FNAME+' \n')
  model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
  print('Done!\n')

    # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('/home/grupo01/mcv/models/Sergio/size_' + str(IMAGE_SIZE) + '_BATCH_' + str(batch) + '_accuracy.jpg')
  plt.close()
    # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('/home/grupo01/mcv/models/Sergio/size_' + str(IMAGE_SIZE) + '_BATCH_' + str(batch) +  '_loss.jpg')
  plt.close()

  #to get the output of a given layer
  #crop the model up to a certain layer
  model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

  #get the features from images
  directory = DATASET_DIR+'/test/coast'
  x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0] )))
  x = np.expand_dims(np.resize(x, (IMAGE_SIZE, IMAGE_SIZE, 3)), axis=0)
  print('prediction for image ' + os.path.join(directory, os.listdir(directory)[0] ))
  features = model_layer.predict(x/255.0)
  print(train_features)
  print('Done!')

  #use svm with the output of second layer
  
  #PSEUDOCODE 
  svm = applySVM(train_features,train_labels)
  svm_pred = svm.predict(test_features, test_labels)
  svm_acc = accuracy_score(validation_generator.classes, svm_pred)
  print("Accuracy of SVM-rbf: %.2f" %(svm_acc*100))
