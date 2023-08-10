import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(
    page_title="Train your model",
)
st.title("Train your model")

def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

def get_patches_from_csv_path(csv_path):
    h5_files = np.array(csv["Path"].values)
    h5_filename = np.array(csv["WSI_name"].values)
    h5_labels = np.array(csv["Label"].values)
    total_iterations = len(h5_files)
    progress_bar = st.progress(0)
    patches = []
    labels = []
    patches_count = 0
    for (i, h5_file) in enumerate(h5_files):
        with h5py.File(h5_file, 'r') as data:
            number_keys = list(data.keys())
            patches_count += len(number_keys)
            if h5_labels[i] == "Tumor":
                labels.append(([0] * len(number_keys)))
            if h5_labels[i] == "Control":
                labels.append(([1] * len(number_keys)))
            for key in number_keys:
                patch = data[key][:]
                patches.append(patch)

        progress = (i + 1) / total_iterations
        st.write(f"Found {len(number_keys)} patches for the image {h5_filename[i]}")
        progress_bar.progress(progress, text="Reading patches... Please wait.")

    patches = np.array(patches)
    labels = np.concatenate(labels)
    labels = np.eye(2)[labels]
    #labels = labels.reshape(labels.shape[1], labels.shape[2])

    return patches, labels

csv_file = st.file_uploader("Cargar archivo CSV", type=['csv'])

if csv_file is not None:
    csv = load_csv(csv_file)

    # Mostrar el DataFrame
    st.write("Contenido del archivo CSV:")
    st.dataframe(csv)

    # Seleccionar la columna con directorios .h5
    #h5_column = st.selectbox("Seleccionar columna con directorios .h5", df.columns)
    start_training = st.button("Start training")
    if start_training:
        st.snow()
        patches, labels = get_patches_from_csv_path(csv)
        train_data, test_data, train_labels, test_labels = train_test_split(patches, labels, test_size=0.2, random_state=42)
        train_data = train_data[:20]
        train_labels = train_labels[:20]
        test_data = test_data[20:25]
        test_labels = test_labels[20:25]

        st.success("Your data has been loaded into the platform.")



        # Construir el modelo de TensorFlow

        data_augmentation = keras.Sequential(
            [
                # Reflejo horizontal
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                # Rotaciones
                layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )

        # base VGG16
        base_model = keras.applications.VGG16(
            weights="imagenet",  # Cargamos los pesos entrenados con ImageNet.
            input_shape=(256, 256, 3),  # por defecto acepta 299x299x3
            include_top=False,  # La primera capa  no se incluye
        )

        # Congelamos
        # Congelamos
        base_model.trainable = False

        # Añadirmos la primera capa
        inputs = keras.Input(shape=(256, 256, 3))
        x = data_augmentation(inputs)  # Aplicamos el aumento de datos.
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)  # Regularizamos
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.categorical_accuracy],
        )
        with st.spinner('Training your model, please wait:'):
            history = model.fit(train_data, train_labels, epochs=10, verbose=1)

        st.success("The data has been trained! Here there are your results:")

        # Visualizar la loss y el accuracy
        st.subheader("Gráficas de Loss y Accuracy")

        # Loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='train_loss')
        # plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['categorical_accuracy'], label='train_acc')
        # plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        st.pyplot(plt)

        # Evaluar el modelo en los datos de prueba
        test_loss, test_acc = model.evaluate(test_data, test_labels)
        st.subheader("Métricas en datos de prueba")
        st.write(f"Loss: {test_loss:.4f}")
        st.write(f"Accuracy: {test_acc:.4f}")


#get_patches_from_csv_path()