# -*- coding: utf-8 -*-
"""
# Detecting Benign/Malignant Skin Cancer

---
"""


# Import dependencies
import os
import subprocess
import sys
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import pathlib
import pip

# Unzip the data file
zip_file = zipfile.ZipFile("data.zip")
zip_file.extractall()
zip_file.close()


# Display contents in each directory
for dirpath, dirnames, filenames in os.walk("train"):
  print(f"{len(dirnames)} directories; {len(filenames)} files in '{dirpath}'.")
for dirpath, dirnames, filenames in os.walk("test"):
  print(f"{len(dirnames)} directories; {len(filenames)} files in '{dirpath}'.")


# Install correct NumPy version
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
def install_numpy(version="1.24.3"):
    try:
        # Run pip command to install the specific version of numpy
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"numpy=={version}"])
        print(f"Successfully installed NumPy {version}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install NumPy {version}. Error: {e}")

# Specify the version to install
install_numpy("1.24.3")

# Create class names
data_dir = pathlib.Path("train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)


"""## Preprocess the data"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the random seed
tf.random.set_seed(31)

# Normalize the data
train_norm = ImageDataGenerator(rescale=1./255)
valid_norm = ImageDataGenerator(rescale=1./255)

# Set up paths to our data directories
train_dir = "train"
test_dir = "test"

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_data = train_aug.flow_from_directory(
    train_dir,
    batch_size=32,
    target_size=(224, 224),
    class_mode="binary",
    seed=31
)

# Import data from directories and turn it into batches
# train_data = train_norm.flow_from_directory(train_dir,
#                                                batch_size=32,
#                                                target_size=(224, 224),
#                                                class_mode="binary",
#                                                seed=31)
valid_data = valid_norm.flow_from_directory(test_dir,
                                              batch_size=32,
                                              target_size=(224, 224),
                                              class_mode="binary",
                                              seed=31)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the MobileNetV2 model with pretrained weights
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

from tensorflow.keras.layers import Dropout
# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

# Create the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=0.0001),
              metrics=["accuracy"])


# Predictions
y_pred = model.predict(valid_data)
y_pred = (y_pred > 0.5).astype(int)


from sklearn.metrics import classification_report

print(classification_report(
    valid_data.classes,
    y_pred,
    target_names=class_names,
    zero_division=0
))

from tensorflow.keras.callbacks import Callback

# Predictions callback
class PredictionLogger(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions on the validation data
        val_images, val_labels = self.validation_data.next()
        val_preds = (self.model.predict(val_images) > 0.5).astype(int).flatten()
        
        # Count how often each class was predicted
        unique, counts = np.unique(val_preds, return_counts=True)
        prediction_counts = dict(zip(unique, counts))
        
        print(f"\nEpoch {epoch + 1} predictions:")
        print(f"Class 0 (benign): {prediction_counts.get(0, 0)}")
        print(f"Class 1 (malignant): {prediction_counts.get(1, 0)}")

# Get validation data generator
valid_data.reset()
callbacks = [PredictionLogger(validation_data=valid_data)]

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)
callbacks.append(early_stopping)


from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5, 
    patience=3, 
    verbose=1
)
callbacks.append(lr_scheduler)

# Fit the model
history = model.fit(train_data,
                        epochs=25,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data),
                        callbacks=callbacks)


# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()

# Save the model after training
model.save("skin_cancer_model.h5")

# Load the model for predictions
from tensorflow.keras.models import load_model
model = load_model("skin_cancer_model.h5")