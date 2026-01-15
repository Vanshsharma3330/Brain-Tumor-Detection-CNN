# import warnings
# warnings.filterwarnings('ignore')


# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import math


# # Count number of images in each class (Training Set)
# ROOT_DIR = "dataset/Training"
# number_of_images = {}
# for folder in os.listdir(ROOT_DIR):
#     folder_path = os.path.join(ROOT_DIR, folder)
#     number_of_images[folder] = len(os.listdir(folder_path))

# print(number_of_images)


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32

# train_val_datagen = ImageDataGenerator(
#     rescale = 1./255,
#     validation_split = 0.176
# )
# test_datagen = ImageDataGenerator(
#     rescale = 1./255
# )

# #The dataset already provided separate training and testing folders. I further split the training set using Keras’ validation split to obtain a 70% training, 15% validation, and 15% testing distribution without physically moving files, ensuring no data leakage.

# train_generator = train_val_datagen.flow_from_directory(
#     "dataset/Training",
#     target_size = IMG_SIZE,
#     batch_size = BATCH_SIZE,
#     class_mode = "categorical",
#     subset = "training",
#     shuffle = True
# )

# val_generator = train_val_datagen.flow_from_directory(
#     "dataset/Training",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="validation",
#     shuffle=True
# )

# test_generator = test_datagen.flow_from_directory(
#     "dataset/Testing",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     shuffle=False
# )


# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
# from tensorflow.keras.models import Sequential


# # Build CNN model
# model = Sequential()

# # Block 1
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Block 2
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Block 3
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Regularization
# model.add(Dropout(0.25))

# # Classification head
# model.add(GlobalAvgPool2D())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))  # 4 classes

# # Model summary
# model.summary()

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )


# from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint

# early_stop = EarlyStopping(
#     monitor='val_accuracy',
#     min_delta=0.01,
#     patience=3,
#     verbose=1,
#     mode='max',
#     restore_best_weights=True
# )

# model_checkpoint = ModelCheckpoint(
#     filepath='best_model.h5',
#     monitor='val_accuracy',
#     save_best_only=True,
#     verbose=1,
#     mode='max'
# )


# callbacks = [early_stop, model_checkpoint]

# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=30,
#     callbacks=callbacks,
#     verbose=1
# )

# from tensorflow.keras.models import load_model
# best_model = load_model("best_model.h5")

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    BatchNormalization,
    GlobalAvgPool2D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array



ROOT_DIR = "dataset/Training"
number_of_images = {}

for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)
    number_of_images[folder] = len(os.listdir(folder_path))

print("Training data distribution:", number_of_images)



IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.176   # ensures overall 70/15/15 split
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_val_datagen.flow_from_directory(
    "dataset/Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_val_datagen.flow_from_directory(
    "dataset/Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    "dataset/Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)



model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Block 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Block 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Regularization + head
model.add(Dropout(0.25))
model.add(GlobalAvgPool2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # 4 classes

model.summary()




model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)




early_stop = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01,
    patience=3,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)




history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)




plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()




best_model = load_model("best_model.h5")

test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")




# Change path to any MRI image
img_path = "sample_mri.jpg"  # <-- update this

img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = best_model.predict(img_array)
predicted_class = np.argmax(prediction)

class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}

print("Predicted class:", class_labels[predicted_class])
