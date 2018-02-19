# For Data Processing
import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm

# For CNN implementation
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, global_avg_pool
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
import matplotlib.pyplot as plt

training_dir = './data/train'
test_dir = './data/test1'
img_size = 50
learning_rate = 1e-4

model = 'CNN Classifier-{}-{}.model'.format(learning_rate, '6conv-basic')


def get_label(img):
    label = img.split('.')[0]

    if label == 'cat':
        return [1, 0]
    else:
        return [0, 1]


def load_training_data():
    training_data = []

    for img in tqdm(os.listdir(training_dir)):
        label = get_label(img)

        img_path = os.path.join(training_dir,img)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (img_size, img_size))

        training_data.append([np.array(img), np.array(label)])

    shuffle(training_data)

    np.save('training_data.npy', training_data)
    return training_data


def pre_process():
    testing_data = []

    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)

        img_num = img.split('.')[0]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (img_size, img_size))

        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)

    np.save('test_data.npy', testing_data)
    return testing_data


# train_data = load_training_data()
# test_data = pre_process()

train_data = np.load('training_data.npy')
test_data = np.load('test_data.npy')

tf.reset_default_graph()
convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = global_avg_pool(convnet)
convnet = dropout(convnet, 0.3)

convnet = fully_connected(convnet, 100, activation='relu')
convnet = dropout(convnet, 0.2)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='rmsprop', learning_rate=learning_rate,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
test_y = [i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=15,
          validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=str(model))
model.save(str(model))


test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate(test_data[:20]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 5, num + 1)
    orig = img_data
    data = img_data.reshape(img_size, img_size, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()
