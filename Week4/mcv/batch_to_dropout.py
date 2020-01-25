import os
import getpass

from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.models import Model, Sequential
import keras
from keras.layers import Dense, GlobalAveragePooling2D, SpatialDropout2D, SpatialDropout3D, Dropout, Conv2D, Activation, BatchNormalization, ZeroPadding2D, Input, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re

TEST_DATASET_DIR = '/home/mcv/datasets/MIT_split'
TRAIN_DATASET_DIR ='/home/grupo01/mcv/train_dataset'
BASE_PATH = '/home/grupo01/mcv/models/Josep'
NUM_CLASSES = 8
BATCH = 20
OPTIMIZERS = ['SGD']
IMAGE_SIZE = 224  # ResNet50 default
DATAGEN = ImageDataGenerator(preprocessing_function=preprocess_input)
TEST_DATAGEN = ImageDataGenerator(preprocessing_function=preprocess_input)

START_OF_SUBLAYER_INDEXES = [17]
#PATIENCE_EARLY = [8, 10, 12]
#PATIENCE_REDUCEONPLATEAU = [3, 5, 8]
#NUM_LAYERS = 176

#PATIENCE_EARLY = 10
PATIENCE_REDUCEONPLATEAU = 8
FIRST_CUT = 2
SECOND_CUT = 7
THIRD_CUT = 10
FOURTH_CUT = 13
FIFTH_CUT = 14
LAST_CUT = 17

"""def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)

def insert_layer_factory():
    return Dropout(0.5)"""


"""base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

x = base_model.layers[LAST_CUT].output
x = GlobalAveragePooling2D()(x)
x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
model_def = Model(base_model.input, x)

insert_layer_nonseq(model_def, 'activation_1', insert_layer_factory, position='after')
insert_layer_nonseq(model_def, 'activation_2', insert_layer_factory, position='after')
insert_layer_nonseq(model_def, 'activation_3', insert_layer_factory, position='after')"""

#insert_layer_nonseq(model_def, 'bn2a_branch2c', insert_layer_factory, position='replace')
#insert_layer_nonseq(model_def, 'bn2a_branch2d', insert_layer_factory, position='replace')
#insert_layer_nonseq(model_def, 'bn2a_branch1', insert_layer_factory, position='replace')

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = Dropout(rate=0.5, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = Dropout(rate=0.5, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

path = os.path.join(BASE_PATH, 'REDUCEONPLATEAU_' + str(PATIENCE_REDUCEONPLATEAU) + '_POOLING_AVG_SGD' + str(LAST_CUT) + '_')

# Determine proper input shape

# Determine proper input shape
input_shape = (224, 224, 3)

input_tensor = None
if input_tensor is None:
    img_input = Input(shape=input_shape)
else:
    if not keras.backend.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor
if keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
x = Conv2D(64, (7, 7),
                    strides=(2, 2),
                    padding='valid',
                    kernel_initializer='he_normal',
                    name='conv1')(x)
x = Dropout(rate=0.5, name='bn_conv1')(x)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))


x = GlobalAveragePooling2D()(x)
x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = Model(img_input, x, name='resnet50')
print(model.summary())

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=PATIENCE_REDUCEONPLATEAU, epsilon=1e-04, cooldown=0, min_lr=0)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = DATAGEN.flow_from_directory(
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

history = model.fit_generator(
            train_generator,
            steps_per_epoch=400 // BATCH,
            epochs=60,
            validation_data=validation_generator,
            validation_steps=807 // BATCH,
            verbose=1,
            callbacks=[reduce_on_plateau])

print('Done!\n')
print('Saving the model\n')
model.save(path + 'model.h5')  # always save your weights after training or during training
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