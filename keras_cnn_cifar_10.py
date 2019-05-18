# https://blog.csdn.net/yph001/article/details/82950570
import numpy as np
import time

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, Add, Activation, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D


from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils, plot_model
from keras import backend
from keras import backend as K
from keras.regularizers import l2


weight_decay = 0.0005


def expand_conv(init, base, k, stride):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    shortcut = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    shortcut = Activation('relu')(shortcut)

    x = ZeroPadding2D((1, 1))(shortcut)
    x = Convolution2D(base * k, (3, 3), strides=stride, padding='valid', kernel_initializer='he_normal',
                      use_bias=False)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(base * k, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal',
                      use_bias=False)(x)

    shortcut = Convolution2D(base * k, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal',
                             use_bias=False)(shortcut)

    m = Add()([x, shortcut])

    return m


def conv_block(input, n, stride, k=1, dropout=0.0):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(n * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(n * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    m = Add()([init, x])
    return m


def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Wide Residual Network with specified parameters
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    ip = Input(shape=input_dim)
    x = ZeroPadding2D((1, 1))(ip)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    nb_conv = 4
    x = expand_conv(x, 16, k, stride=(1, 1))
    for i in range(N - 1):
        x = conv_block(x, n=16, stride=(1, 1), k=k, dropout=dropout)
        nb_conv += 2

    x = expand_conv(x, 32, k, stride=(2, 2))

    for i in range(N - 1):
        x = conv_block(x, n=32, stride=(2, 2), k=k, dropout=dropout)
        nb_conv += 2

    x = expand_conv(x, 64, k, stride=(2, 2))

    for i in range(N - 1):
        x = conv_block(x, n=64, stride=(2, 2), k=k, dropout=dropout)
        nb_conv += 2

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)
    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


backend.set_image_data_format('channels_first')

# 设定随机数种子
seed = 7
np.random.seed(seed)

def unPickle(file):
    import pickle as pk
    with open(file, 'rb') as f:
        d = pk.load(f, encoding='bytes')
    return d

def read_cifar_10_from_file(file_name):
    print('in read_cifar_10_from_file')
    data = unPickle(file_name)
    img = data[b'data']
    print('img.shape is', img.shape)  # 显示为（10000，3072）

    img_0 = img[0] #得到第一张图像
    img_reshape = img_0.reshape(3,32,32)
    import PIL.Image as image
    import matplotlib.pyplot as plt
    r = image.fromarray(img_reshape[0]).convert('L')
    g = image.fromarray(img_reshape[1]).convert('L')
    b = image.fromarray(img_reshape[2]).convert('L')
    img_m = image.merge('RGB',(r,g,b))
    plt.imshow(img_m)
    plt.show()


# read_cifar_10_from_file('./data/cifar-10-python/cifar-10-batches-py/data_batch_1')


def batcher(X_data, y_data, batch_size=-1, random_seed=None):
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    if random_seed is None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(random_seed)

    rnd_idx = np.random.permutation(len(X_data))
    # print('rnd_idx[:10] is', rnd_idx[:10])
    n_batches = len(X_data) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X_data[batch_idx], y_data[batch_idx]
        yield X_batch, y_batch


# 导入数据
(x_train, y_train), (x_validation, y_validation) = cifar10.load_data()
x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_train = x_train/255.0
x_validation = x_validation/255.0
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]


def create_deeper_model(epochs=25):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='elu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())

    # model.add(Conv2D(256, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    # model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())

    # model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1024, activation='elu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())

    model.add(Dense(512, activation='elu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())

    model.add(Dense(512, activation='elu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())

    model.add(Dense(256, activation='elu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax'))
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_deep_model(epochs=25):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='elu', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='elu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='elu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='elu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())


    model.add(Dense(10, activation='softmax'))
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model(epochs=25):
    print('in Create_model')
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# 训练模型及评估模型
# model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=32, verbose=2)

print('x_train.shape is ', x_train.shape, 'y_train.shape is ', y_train.shape,
      'x_validation.shape is ', x_validation.shape, 'y_validation.shape is ', y_validation.shape)

def keras_cnn_test(x_data, y_data):
    epochs = 25
    # model = create_model(epochs)
    # model = create_deep_model()
    model = create_deeper_model()

    total_iter_num = 0
    max_epochs = 100
    batch_size = 32
    for epoch in range(max_epochs):
        print('current epoch is ', epoch)
        epoch_start_t = time.time()
        for batch_X, batch_y in batcher(x_train, y_train, batch_size):
            total_iter_num += 1
            model.train_on_batch(batch_X, batch_y)
            if total_iter_num%100==0:
                train_score = model.evaluate(x=batch_X, y=batch_y, verbose=0)
                val_score = model.evaluate(x=x_validation, y=y_validation, verbose=0)
                print('epoch:{:3d} total_iter_num:{:6d}, Train Loss:{:.13f} Accuracy:{:.7f}, Val Loss:{:.13f} Accuracy: {:.7f}'.format(
                    epoch, total_iter_num, train_score[0], train_score[1], val_score[0], val_score[1])
                )
        print('epoch: ', epoch, 'cost time', time.time()-epoch_start_t)


if __name__=='__main__':
    print('start training!')
    keras_cnn_test(x_train, y_train)

    # init = (3, 32, 32)
    # wrn_28_10 = create_wide_residual_network(init, nb_classes=10, N=4, k=10, dropout=0.0)

    # model = create_deeper_model()
    # model.summary()
    # plot_model(model, "./WRN-28-8.png", show_shapes=True)





