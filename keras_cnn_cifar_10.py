# https://blog.csdn.net/yph001/article/details/82950570


from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend
backend.set_image_data_format('channels_first')

# 设定随机数种子
seed = 7
np.random.seed(seed)


# 导入数据
(x_train, y_train), (x_validation, y_validation) = cifar10.load_data()
x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_train = x_train/255.0
x_validation = x_validation/255.0

# 进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]


# 定义模型创建函数
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
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


epochs = 25
model = create_model(epochs)

print('start training!')

# 训练模型及评估模型
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=32, verbose=2)
score = model.evaluate(x=x_validation, y=y_validation, verbose=0)
print('Accuracy: %.2f%%' % (score[1] * 100))
