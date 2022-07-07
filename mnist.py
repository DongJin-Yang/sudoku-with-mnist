from keras.datasets import mnist
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
import tensorflow as tf

batch_size = 128
classes = 10
epochs = 10

# 데이터 가져오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 네트워크 구성
model = models.Sequential()
model.add(Conv2D(filters=64, 
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                input_shape=(28, 28, 1)))
model.add(Conv2D(filters=64, 
                 kernel_size=(3, 3), 
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# MNIST 이미지 훈련
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = train_images.astype('float32')/255

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.optimizers.Adam(lr=0.001), 
              metrics=['accuracy'])
model.summary()

model.fit(train_images, train_labels, 
          epochs=10, 
          batch_size=100,
         verbose=1,
         validation_data=(test_images, test_labels))

# 평가
score = model.evaluate(test_images, test_labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 저장
model.save('model.h5')
model.save_weights('weights.h5')