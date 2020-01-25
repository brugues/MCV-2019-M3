import os
import getpass

from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import random 

    
def Rand(start, end, num): 
    res = [] 
  
    for j in range(num): 
        res.append(random.randint(start, end)) 
  
    return res 
    
    
def createModel(optimizer='adam'):

  # Parameters
  NUM_CLASSES = 8
 
  START_OF_SUBLAYER_INDEX = 17
  
  model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
  
  x = model.layers[START_OF_SUBLAYER_INDEX].output
  x = GlobalAveragePooling2D()(x)
  x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
  model_def = Model(model.input,x)
  
  reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, epsilon=1e-04, cooldown=0, min_lr=0)
  model_def.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  
  return model_def


def main():

  BASE_PATH = '/home/grupo01/mcv/models/Sergio'
  TEST_DATASET_DIR = '/home/mcv/datasets/MIT_split'
  TRAIN_DATASET_DIR ='/home/grupo01/mcv/train_dataset'
  IMAGE_SIZE = 224  # ResNet50 default
  TRAIN_DATAGEN = ImageDataGenerator(preprocessing_function=preprocess_input)
  #TEST_DATAGEN = ImageDataGenerator(preprocessing_function=preprocess_input)
  
  path = os.path.join(BASE_PATH, 'RANDOM_HYPERPARAMETER_OPTIMIZATION_' + str(17) + '_')
  
  train_generator = TRAIN_DATAGEN.flow_from_directory(
      TRAIN_DATASET_DIR,  # this is the target directory
      target_size=(IMAGE_SIZE, IMAGE_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
      batch_size=16,
      classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
      class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
  data = []    
  labels = []   
  max_iter = 400  
  i = 0
  for d, l in train_generator:
    data.append(d)
    labels.append(l)
    i += 1
    if i == max_iter:
        break

  data = np.array(data)
  data = np.reshape(data, (data.shape[0]*data.shape[1],) + data.shape[2:])

  labels = np.array(labels)
  labels = np.reshape(labels, (labels.shape[0]*labels.shape[1],) + labels.shape[2:])
  
  #validation_generator = TRAIN_DATAGEN.flow_from_directory(
  #    TEST_DATASET_DIR+'/test',
  #    target_size=(IMAGE_SIZE, IMAGE_SIZE),
  #    batch_size=batch_size,
  #    classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  #    class_mode='categorical')
  
  # Find best parameters
  param_grid = {'batch_size':               Rand(8, 256, 10), 
                'optimizer':                ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}
                #'learning_rate':            [0.0001, 0.001, 0.01, 0.1], 
                #'patience_reduceonplateau': Rand(3,15,5)}
                  
  model = KerasClassifier(build_fn=createModel, verbose=0)
  print('Keras model created')
  randomizedsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5)
  print('Random search done!')
  best_model_random = randomizedsearch.fit(data, labels)
  
  print("Best: %f using %s" % (best_model_random.best_score_, best_model_random.best_params_))
  means = best_model_random.cv_results_['mean_test_score']
  stds = best_model_random.cv_results_['std_test_score']
  params = best_model_random.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))
  
  
  #history = model_def.fit_generator(
  #            train_generator,
  #            steps_per_epoch=400 // BATCH,
  #            epochs=60,
  #            validation_data=validation_generator,
  #            validation_steps=807 // BATCH,
  #            verbose=1,
  #            callbacks=[reduce_on_plateau])
  
  print('Done!\n')

if __name__ == "__main__":
  main()

