from keras.models import Sequential
from keras import regularizers, optimizers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, TimeDistributed, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D

__version__='1.0'


def cnn1(input_shape, out_classes=10):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(4, 4), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(rate=0.35))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(rate=0.35))
    model.add(Flatten())
    model.add(Dense(units=1024))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=out_classes))
    model.add(Activation('tanh'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn2(input_shape, out_classes=10):
    model = Sequential()
    model.add(Conv2D(32,
                     (3, 3),
                     padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_classes, activation='softmax'))
    model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    return model

def cnn3(input_shape, out_classes=10):
    
    model = Sequential()
        
    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=input_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(out_classes, activation='relu'))
    
    model.compile(optimizers.Adam(0.0002, 0.5), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def cnn4(input_shape, out_classes=10):
    ''' creating Sequential model '''
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(out_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model
 
def cnn5(input_shape, out_classes=10):
    model = Sequential()
  
    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(out_classes))
    model.add(Activation('softmax'))
    model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=['accuracy'])
    return model
