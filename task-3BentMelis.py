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

image_path = "resources/data/train_set/drumset_white_background/drumset white background_image_15.jpg"
image = Image.open(image_path)

class_names = ["drumset", "guitar", "piano", "saxophone", "violin"]
st.title("Taak DL - Bent Melis (r0831245)")

st.text("Laten we beginnen met een kijkje te nemen naar de data, hier is een van de images:")
image_width = 300
st.image(image, width=image_width, caption="Drumset Image")

epochs = st.slider("Selecter het aantal epochs:", min_value=1, max_value=7, value=8)

st.text("Helaas heb ik het aantal epochs moeten limiteren op 8 voor de Streamlit versie.")
st.text("Omdat streamlit 'healthchecks' stuurt naar de applicaties,")
st.text("crashed de applicatie wanneer het begint met het model te trainen.")
st.text("Daarom gebruik ik hier een model tot max 8 epochs, zodat de essentie wel duidelijk")
st.text("word.")
st.text("")


st.text("Laten we beginnen met de training:")
if st.button("Start Training"):
    st.text("")
    st.text("The training is gestart!")

    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_ds = test_datagen.flow_from_directory('resources/data/train_set/', target_size=(128, 128), batch_size=32, seed=123, class_mode="categorical")
    val_ds = test_datagen.flow_from_directory('resources/data/test_set/', target_size=(128, 128), batch_size=32, seed=123, class_mode='categorical')

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


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(train_ds, steps_per_epoch=30, epochs=epochs, validation_data=val_ds)
    st.text("")
    st.text("")

    test_loss, test_acc = model.evaluate(val_ds)
    def plotLosses(history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    st.text("Laten we kijken of de testen goed zijn!")
    st.text(f"Validation accuracy: {test_acc}")
    st.pyplot(plotLosses(history))

    y_pred_probabilities = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probabilities, axis=1)
    y_true = val_ds.classes

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt)