import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 

train_dir = r"D:\SignLanguageTranslator\Data\Train"
validation_dir = r"D:\SignLanguageTranslator\Data\Validation"
model_save_path = r"D:\SignLanguageTranslator\Model\trained_model.h5"

img_size = (128, 128)
batch_size = 32 
epochs = 20  
num_classes = len(os.listdir(train_dir))

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,  
    width_shift_range=0.3, 
    height_shift_range=0.3, 
    shear_range=0.3,  
    zoom_range=0.3, 
    horizontal_flip=True,
    fill_mode='nearest' 
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)  
x = Dropout(0.5)(x)  
x = Dense(256, activation="relu")(x) 
x = Dropout(0.3)(x) 
output_layer = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.00001), loss="categorical_crossentropy", metrics=["accuracy"])

fine_tuning_epochs = 10

history_fine = model.fit(
    train_generator,
    epochs=fine_tuning_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print("Model saved!")

def plot_training(history, history_fine):
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

plot_training(history, history_fine)