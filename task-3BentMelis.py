#ik leg de library toe als ik hem gebruik in de code!
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , optimizers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import MaxNorm
from keras import utils
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import streamlit as st


#hier plot ik een image met de PIL library en de matplotlib library
# Load the image
image_path = "resources/data/train_set/drumset_white_background/drumset white background_image_15.jpg"
image = Image.open(image_path)

# Display the image in Streamlit
st.image(image, use_column_width=True)  # 'use_column_width=True' adjusts the image width to the column width

# Optionally, you can hide the image's filename and adjust the layout
st.markdown("### Drumset Image")  # Optional title or description
st.text("")  # Optional space between title/description and the image


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
print("trainingset:")
train_ds = test_datagen.flow_from_directory('resources/data/train_set/',
                                            target_size=(128,128),
                                            batch_size=32,
                                            # subset="training",
                                            seed=123,
                                            class_mode="categorical")
print("testset:")
val_ds = test_datagen.flow_from_directory('resources/data/test_set/',
                                            target_size=(128,128),
                                            batch_size=32,
                                            #subset="training",
                                            seed=123,
                                            class_mode='categorical')

class_names = ["drumset", "guitar", "piano", "saxophone", "violin"]

for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    if i < 10:
        img_data_batch, label_batch = next(train_ds)
        
        img_data = img_data_batch[0]
        label = label_batch[0]
        
        plt.imshow(img_data, cmap='gray')
        
        label_index = np.argmax(label)
        
        label_name = class_names[label_index]
        
        label = plt.xlabel("Train: " + label_name).set_color("green")
    else:
        img_data_batch, label_batch = next(val_ds)
        
        img_data = img_data_batch[0]
        label = label_batch[0]
        
        plt.imshow(img_data, cmap='gray')
        
        label_index = np.argmax(label)
        
        label_name = class_names[label_index]
        
        label = plt.xlabel("Test: " + label_name).set_color("red")

plt.show()

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(16,16,3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(32,activation="relu"))

model.add(Dense(5,activation="softmax"))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())

history = model.fit_generator(train_ds,
                    steps_per_epoch = 30,
                    epochs = 20,
                    validation_data = val_ds)

def plotLosses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
plotLosses(history)

test_loss, test_acc = model.evaluate(val_ds)
print("Validation accuracy:", test_acc)

# Make predictions on the validation set
y_pred_probabilities = model.predict(val_ds)
y_pred = np.argmax(y_pred_probabilities, axis=1)
y_true = val_ds.classes

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()