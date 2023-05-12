import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import numpy as np

import os

# 設置訓練和驗證數據路徑
train_data_path = "D:/sexual exploitation/train data"
valid_data_path = "D:/sexual exploitation/valid data"

# 設置圖像尺寸和批量大小
img_width, img_height = 150, 150
batch_size = 16

# 創建 ImageDataGenerator 用於數據擴充
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,  # 將 shear_range 設置為 0.1
    zoom_range=0.1,   # 將 zoom_range 設置為 0.1
    horizontal_flip=True)


valid_datagen = ImageDataGenerator(rescale=1./255)

# 定義模型
model = Sequential([
    keras.Input(shape=(img_width, img_height, 3)),  # 新增一個輸入層，指定圖像尺寸和通道數
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# # 印出圖片(檢查用))
# files = os.listdir(train_data_path)
# for file in files:
#     print(os.path.join(train_data_path, file))

# 加載數據集 A
data_A = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/sexual exploitation/A", batch_size=batch_size, image_size=(img_width, img_height))

# 加載數據集 B
data_B = tf.keras.preprocessing.image_dataset_from_directory(
    'D:/sexual exploitation/B', batch_size=batch_size, image_size=(img_width, img_height))

# 訓練集 A 中的圖像數量
num_images_A = data_A.cardinality().numpy() * batch_size

# 訓練集 B 中的圖像數量
num_images_B = data_B.cardinality().numpy() * batch_size

# 生成訓練和驗證數據
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
    valid_data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# 計算步數
steps_per_epoch = max(num_images_A, num_images_B) // batch_size

# 整合數據集
data = data_A.concatenate(data_B)

# 建立模型
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 編譯模型
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# 訓練模型
history = model.fit(
    data,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size)

#評估模型
test_loss, test_acc = model.evaluate(valid_generator, steps=valid_generator.samples // batch_size)
print('Test accuracy:', test_acc)

#保存模型
model.save('D:/sexual exploitation/my_model.h5')
