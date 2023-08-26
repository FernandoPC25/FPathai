
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import time
import random

st.set_page_config(
    page_title="Train your model",
)
st.title("Train your model")
st.write("Explain the process...")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(predicted_labels, true_labels):
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    return f1, precision, recall


def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

# def get_patches_from_csv_path(csv_path):
#     h5_files = np.array(csv["Path"].values)
#     h5_filename = np.array(csv["WSI_name"].values)
#
#     # Obtain the labels and one hot them
#     h5_labels = np.array(csv["Label"].values)
#     label_mapping = {label: idx for idx, label in enumerate(set(h5_labels))}
#     num_classes = len(label_mapping)
#     labels_encoded = [label_mapping[label] for label in h5_labels]
#     labels_encoded = np.eye(num_classes)[labels_encoded].astype(int)
#
#
#     total_iterations = len(h5_files)
#     progress_bar = st.progress(0)
#     patches = []
#     labels = []
#     #patches_count = 0
#     for (i, h5_file) in enumerate(h5_files):
#         with h5py.File(h5_file, 'r') as data:
#             number_keys = list(data.keys())
#             #patches_count += len(number_keys)
#             # if h5_labels[i] == "Tumor":
#             #     labels.append(([0] * len(number_keys)))
#             # if h5_labels[i] == "Control":
#             #     labels.append(([1] * len(number_keys)))
#             for key in number_keys:
#                 patch = data[key][:]
#                 patches.append(patch)
#             labels.append(np.array([labels_encoded[i] for _ in range(len(number_keys))]))
#
#         progress = (i + 1) / total_iterations
#         st.write(f"Found {len(number_keys)} patches for the image {h5_filename[i]}")
#         progress_bar.progress(progress, text="Reading patches... Please wait.")
#
#     patches = np.array(patches)
#     labels = np.concatenate(labels, axis=0)
#     #labels = np.eye(2)[labels]
#     #labels = labels.reshape(labels.shape[1], labels.shape[2])
#
#     return patches, labels, num_classes


def training_metrics(results, test_csv):

    #plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.plot(results['categorical_accuracy'], label='Accuracy')
    if val:
        plt.plot(results['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([-1, epochs])
    plt.grid()
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.plot(results['loss'], label='Loss')
    if val:
        plt.plot(results['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([-1, epochs])
    plt.grid()
    plt.legend(loc='best')
    #plt.tight_layout()
    st.pyplot(plt)

    chart_data = pd.DataFrame(
        data={"Training Accuracy": results['categorical_accuracy'],
              "Epochs": list(range(1, epochs+1))
              }
        )

    if val:
        chart_data = pd.DataFrame(
            data={"Training Accuracy": results['categorical_accuracy'],
                  "Validation Accuracy": results['val_categorical_accuracy'],
                  "Epochs": list(range(1, epochs+1))
                  }

        )


    st.line_chart(chart_data, x="Epochs")
    #


    # Evaluar el modelo en los datos de prueba

    test_data = read_patches(test_csv)[0]
    test_labels = read_patches(test_csv)[1]

    test_loss, test_acc = model.evaluate(test_data, test_labels)


    predictions = model.predict(test_data)
    conf_matrix = confusion_matrix(test_labels.argmax(-1), predictions.argmax(-1))

    f1, precision, recall = compute_metrics(test_labels.argmax(-1), predictions.argmax(-1))

    # precision = precision_score(test_labels.argmax(-1), predictions.argmax(-1))
    # recall = recall_score(test_labels.argmax(-1), predictions.argmax(-1))
    # f1 = f1_score(test_labels.argmax(-1), predictions.argmax(-1))
    import seaborn as sns

    plt.figure(figsize=(10, 8))

    #cbar_ksw ={'format':'%.0f'}

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, cbar_kws= {'format':'%.0f'},
                xticklabels=list(label_mapping.keys()), yticklabels=list(label_mapping.keys())
                #,vmin=np.min(conf_matrix), vmax=np.max(conf_matrix)
                )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label', rotation=90)
    plt.title('Confusion Matrix')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    #plt.xtics(list(label_mapping.keys()), rotation=45)
    #plt.ytics(list(label_mapping.keys()), rotation=90)

    st.pyplot(plt)

    # Sample data
    data = {
        'Metric': ['Loss', 'Accuracy', 'F1-Score', 'Precision', 'Recall'],
        'Value': [test_loss, test_acc, f1, precision, recall]
    }

    df_results = pd.DataFrame(data)
    df_results.set_index('Metric', inplace=True)
    st.table(df_results)


    # # ROC AUC
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import roc_curve
    #
    # # Calculate the AUC-ROC score
    #
    #
    # if test_labels.shape(1) == 2:
    #     auc_roc = roc_auc_score(test_labels.argmax(-1), predictions.argmax(-1))
    #     fpr, tpr, _ = roc_curve(test_labels.argmax(-1), predictions.argmax(-1))
    # # else:
    # #     auc_roc = []
    # #     fpr = []
    # #     tpr = []
    # #     for i in range(num_classes):
    # #         auc_roc.append(roc_auc_score(true_labels[:, i], predicted_probabilities[:, i]))
    # #         fpr_i, tpr_i, _ = roc_curve(true_labels[:, i], predicted_probabilities[:, i])
    # #         fpr.append(fpr_i)
    # #         tpr.append(tpr_i)
    #
    #
    # # Plot the ROC curve
    # plt.figure(figsize=(10, 8))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")
    # st.pyplot(plt)


def create_csv_of_random_keys(csv_full, num_patches_selected):
    h5_filepaths = np.array(csv_full["Path"].values)
    data_with_keys = []
    for (i, h5_filepath) in enumerate(h5_filepaths):
        patient_id = csv_full["Patient_ID"][i]
        label = csv_full["Label"][i]
        with h5py.File(h5_filepath, 'r') as h5_patch:
            total_keys = list(h5_patch.keys())
            selected_keys = random.sample(total_keys, min(num_patches_selected, len(total_keys)))
            for key in selected_keys:
                data_with_keys.append({"Path": h5_filepath, "Key": key, "Patiend_ID": patient_id, "Label": label})

    return pd.DataFrame(data_with_keys)

def get_patch(h5_filepath, key):
    patch = h5py.File(h5_filepath)[key][:]
    return patch

# TODO: mirar si se puede poner aqui read_patches(df_batch, label_mapping)
def read_patches(df_batch):
    paths = df_batch["Path"].tolist()
    keys = df_batch["Key"].tolist()
    labels = df_batch["Label"].tolist()
    data = []
    labels_encoded = [label_mapping[label] for label in labels]
    labels_encoded = np.eye(num_classes)[labels_encoded].astype(int)

    for patch in range(df_batch.shape[0]):
        data.append(get_patch(paths[patch], keys[patch]))

    data = np.array(data)
    return data, labels_encoded

def train_csv_with_keys(csv_with_keys, batch_size, epochs):
    import time
    train_csv, test_csv = train_test_split(csv_with_keys, test_size=0.2, random_state=42)

    if val:
        train_csv, val_csv = train_test_split(train_csv, test_size=0.2, random_state=42)

    start_time = time.time()
    with st.spinner('Training your model, please wait:'):
        if val:
            val_data = read_patches(val_csv)[0]
            val_labels = read_patches(val_csv)[1]
            #print(f"AAAAAAAAAAA val_labels: {val_labels}")
            results = {'loss': [], 'categorical_accuracy': [],
                       'val_loss': [], 'val_categorical_accuracy': []}
            for epoch in range(epochs):
                train_csv = train_csv.sample(frac=1)
                for batch in range(0, train_csv.shape[0], batch_size):
                    df_batch = train_csv.iloc[batch:batch + batch_size]
                    patches, labels = read_patches(df_batch)
                    history = model.fit(patches, labels, verbose=0)
                    print(f"{batch}: {history.history}")

                loss = history.history['loss'][0]
                accuracy = history.history['categorical_accuracy'][0]
                val_loss, val_accuracy = model.evaluate(val_data, val_labels)

                results['loss'].append(loss)
                results['val_loss'].append(val_loss)
                results['categorical_accuracy'].append(accuracy)
                results['val_categorical_accuracy'].append(val_accuracy)
                st.write(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}'
                         f' - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}')

            training_metrics(results, test_csv)

        else:
            results = {'loss': [], 'categorical_accuracy': []}
            for epoch in range(epochs):
                train_csv = train_csv.sample(frac=1)
                for batch in range(0, train_csv.shape[0], batch_size):
                    #st.write(f"batch {batch}")
                    #print(f"batch {batch}")
                    df_batch = train_csv.iloc[batch:batch + batch_size]
                    # print(df_batch)
                    patches, labels = read_patches(df_batch)
                    # print(f"patches.shape: {patches.shape}")
                    # print(f"labels.shape: {labels.shape}")
                    # history = model.fit(train_data, train_labels, batch_size=int(batch_size),
                    #                     validation_data=(validation_data, validation_labels))
                    history = model.fit(patches, labels, verbose=0)
                    #st.write(f'Batch {batch}/{csv_with_keys.shape[0]} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')
                loss = history.history['loss'][0]
                accuracy = history.history['categorical_accuracy'][0]
                results['loss'].append(loss)
                results['categorical_accuracy'].append(accuracy)
                print(f"Metrics in the epoch {epoch}: {history.history}")
                st.write(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')
            training_metrics(results, test_csv)
        time = time.time() - start_time

        st.write(f"Training finished in {time:.4f} seg")



st.write("## Load your data")
csv_file = st.file_uploader("Load the CSV file", type=['csv'])

if csv_file is not None:
    csv = load_csv(csv_file)

    # Show DataFrame
    st.write("Your CSV file:")
    st.dataframe(csv)

    # Preprocessing of Labels
    h5_labels = np.array(csv["Label"].values)
    # which label correspond to the class
    label_mapping = {label: idx for idx, label in enumerate(set(h5_labels))}
    print("label_mapping: ",label_mapping)

    # number of classes
    num_classes = len(label_mapping)
    # labels_encoded = [label_mapping[label] for label in h5_labels]

    number_of_patches = st.slider("Select a number of patches:", 1, 2000, step=1, value=10,
                                  help="How many patches do you want to use for the training")

    csv_with_keys = create_csv_of_random_keys(csv, number_of_patches)

    st.write("------")
    st.write("## Configure your model")

    # Seleccionar la columna con directorios .h5
    #h5_column = st.selectbox("Seleccionar columna con directorios .h5", df.columns)

    choose_a_model = st.selectbox("Select a Transfer Learning model to perform your training",
                                  ("VGG16", "MobileNetV2", "ResNet50", "InceptionV3"),
                                  help="MobileNetV2 is the lightest... blabla"
                                  )
    batch_size = st.slider("Batch Size", 32, 1024, step=16,
                           help="Batch size refers to the quantity of data samples processed in a single forward "
                                "and backward pass by the model during training. This division of input data "
                                "into batches allows the model to update its internal weights after "
                                "handling each batch.")
    epochs = st.slider("Epochs", 1, 100, step=1,
                       help="Epoch refers to a complete pass through the entire training dataset during the training "
                            "phase of a model.")
    choose_optimizer = st.radio("Choose the type of Optimizer ", ("Adam", "SGD", "RMSProp", "Adagrad"),
                                help= "This algorithm is employed to adjust the neural network's weights "
                                      "throughout the training process. The choice of an optimizer "
                                      "shifts the optimization strategy employed by the neural network.")

    learning_rate = st.slider("Learning Rate", 0.0001, float(1),
                             help = "The learning rate serves as a critical hyperparameter within "
                                    "the optimization process of machine learning models. "
                                    "It dictates the magnitude of each step the optimization "
                                    "algorithm takes towards minimizing the loss function.")
    val = st.checkbox('Validation Set',
                      help="The validation data is subset of the available training data."
                           "It is employed to monitor the model's performance and identify potential issues"
                           "such as overfitting.")

    if choose_optimizer == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif choose_optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif choose_optimizer == "RMSProp":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif choose_optimizer == "Adagrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)




    start_training = st.button("Train the model!")
    if start_training:



        st.success("Your configuration is successfully loaded.")

        patch_size = read_patches(csv_with_keys[:1])[0].shape[1:4]

        # Construir el modelo de TensorFlow

        data_augmentation = keras.Sequential(
            [
                # Reflejo horizontal
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                # Rotaciones
                # layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )

        if choose_a_model == "VGG16":
            # base VGG16
            base_model = keras.applications.VGG16(
                weights="imagenet",  # Cargamos los pesos entrenados con ImageNet.
                input_shape=patch_size,  # por defecto acepta 299x299x3
                include_top=False,  # La primera capa  no se incluye
            )

        elif choose_a_model == "MobileNetV2":
            # base MobileNetV2
            base_model = keras.applications.MobileNetV2(
                weights="imagenet",  # Cargamos los pesos entrenados con ImageNet.
                input_shape=patch_size,  # por defecto acepta 299x299x3
                include_top=False,  # La primera capa  no se incluye
            )
        elif choose_a_model == "ResNet50":
            # base ResNet150
            base_model = keras.applications.ResNet50(
                weights="imagenet",
                input_shape=patch_size,
                include_top=False,
            )
        elif choose_a_model == "InceptionV3":
            # base InceptionV3
            base_model = keras.applications.InceptionV3(
                weights="imagenet",
                input_shape=patch_size,
                include_top=False,
            )



        # Congelamos
        base_model.trainable = False

        # Añadirmos la primera capa
        inputs = keras.Input(shape=patch_size)
        x = data_augmentation(inputs)  # Aplicamos el aumento de datos.
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)  # Regularizamos
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.categorical_accuracy],
        )



        train_csv_with_keys(csv_with_keys, batch_size, epochs)




        # steps_per_epoch = len(train_data) // batch_size

        # start_time = time.time()
        # with st.spinner('Training your model, please wait:'):
        #     if val:
        #         results = {'loss': [], 'categorical_accuracy': [],
        #                    'val_loss': [], 'val_categorical_accuracy': []}
        #         for epoch in range(int(epochs)):
        #             # history = model.fit(train_data, train_labels, batch_size=int(batch_size),
        #             #                     validation_data=(validation_data, validation_labels))
        #
        #             history = model.fit(datagen.flow(train_data, train_labels, batch_size=int(batch_size)),
        #                                 validation_data=(validation_data, validation_labels)
        #                                 #, steps_per_epoch=steps_per_epoch,
        #                                 # epochs=epochs
        #                                 )
        #             model.save("prueba.h5")
        #
        #             loss = history.history['loss'][0]
        #             val_loss = history.history['val_loss'][0]
        #             accuracy = history.history['categorical_accuracy'][0]
        #             val_accuracy = history.history['val_categorical_accuracy'][0]
        #             results['loss'].append(loss)
        #             results['val_loss'].append(val_loss)
        #             results['categorical_accuracy'].append(accuracy)
        #             results['val_categorical_accuracy'].append(val_accuracy)
        #             st.write(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}'
        #                      f' - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}')
        #     else:
        #         results = {'loss': [], 'categorical_accuracy': []}
        #         for epoch in range(int(epochs)):
        #             history = model.fit(train_data, train_labels, batch_size=int(batch_size))
        #             loss = history.history['loss'][0]
        #             accuracy = history.history['categorical_accuracy'][0]
        #             results['loss'].append(loss)
        #             results['categorical_accuracy'].append(accuracy)
        #             st.write(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')
        #
        # time = time.time() - start_time
        #
        # st.write(f"Training finished in {time:.4f} seg")

        #st.success("The data has been trained! Here there are your results:")
        st.balloons()

        st.success("Model saved")
        model.save("prueba123456789000.h5")


        # Visualizar la loss y el accuracy
        #st.subheader("Accuracy and Loss graphs")


        #training_metrics()

#get_patches_from_csv_path()