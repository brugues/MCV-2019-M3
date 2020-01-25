import os
import getpass

from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Parameters
TEST_DATASET_DIR = '/home/mcv/datasets/MIT_split'
TRAIN_DATASET_DIR ='/home/grupo01/mcv/train_dataset'
BASE_PATH = '/home/grupo01/mcv/models/Sergio'
NUM_CLASSES = 8
BATCH = 20
OPTIMIZERS = ['SGD']
IMAGE_SIZE = 224  # ResNet50 default
TRAIN_DATAGEN = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=5, width_shift_range=0.1, 
                            height_shift_range=0.1, shear_range=0.1, zoom_range=0.2, fill_mode='nearest', 
                            horizontal_flip=True, vertical_flip=False, rescale=0.8)
TEST_DATAGEN = ImageDataGenerator(preprocessing_function=preprocess_input)

START_OF_SUBLAYER_INDEXES = [17, 27, 37, 49, 59, 69, 79]
NUM_LAYERS = 176

number_of_big_layers = 15
for index in START_OF_SUBLAYER_INDEXES:
    #try:
    path = os.path.join(BASE_PATH, 'POOLING_AVG_SGD_DATA_AUGMENTATION_' + str(index) + '_')

    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    x = model.layers[index].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
    model_def = Model(model.input,x)
  
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model_def.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model_def.summary())


    train_generator = TRAIN_DATAGEN.flow_from_directory(
        TRAIN_DATASET_DIR,  # this is the target directory
        target_size=(IMAGE_SIZE, IMAGE_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH,
        classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    validation_generator = TEST_DATAGEN.flow_from_directory(
        TEST_DATASET_DIR+'/test',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH,
        classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    history = model_def.fit_generator(
                train_generator,
                steps_per_epoch=400 // BATCH,
                epochs=60,
                validation_data=validation_generator,
                validation_steps=807 // BATCH,
                verbose=1)

    print('Done!\n')
    print('Saving the model\n')
    model_def.save(path + 'model.h5')  # always save your weights after training or during training
    print('Done!\n')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path + 'accuracy.jpg')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path + 'loss.jpg')
    plt.close()
    #except:
        #print(str(number_of_big_layers) + 'has crashed')
        #number_of_big_layers -= 1
