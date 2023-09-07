import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import seaborn as sns
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
import time
import random
from sklearn.metrics import roc_auc_score
import uuid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(predicted_labels, true_labels):
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    return f1, precision, recall

def plot_confusion_matrix(conf_matrix, label_mapping):
    plt.figure(figsize=(10, 8))


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

def plot_roc_auc_curve(num_classes, test_labels, predictions, label_mapping):
    classes = list(label_mapping.keys())
    if num_classes == 2:
        plt.figure(figsize=(10, 8))
        auc_score = roc_auc_score(test_labels.argmax(-1), predictions.argmax(-1))
        fpr, tpr, _ = roc_curve(test_labels[:, 1], predictions[:, 1])
        # Plot the ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        st.pyplot(plt)

    else:
        plt.figure(figsize=(10, 8))
        fpr_micro, tpr_micro, _ = roc_curve(test_labels.ravel(), predictions.ravel())
        auc_score = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, label=f'Overall (AUC = {auc_score:.2f})')
        for i in range(num_classes):
            y_true_binary = test_labels[:, i]
            fpr, tpr, _ = roc_curve(y_true_binary, predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        st.pyplot(plt)

    return auc_score


def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocessing_labels(csv):
    h5_labels = np.array(csv["Label"].values)
    label_mapping = {label: idx for idx, label in enumerate(set(h5_labels))}
    num_classes = len(label_mapping)
    return label_mapping, num_classes



def show_training_metrics(results, test_csv):
    if val:
        accuracy_chart = pd.DataFrame(
            data={"Training Accuracy": results['categorical_accuracy'],
                  "Validation Accuracy": results['val_categorical_accuracy'],
                  "Epochs": list(range(1, epochs+1))
                  }
        )

        loss_chart = pd.DataFrame(
            data={"Training Loss": results['loss'],
                  "Validation Loss": results['val_loss'],
                  "Epochs": list(range(1, epochs+1))
                  }
        )

    else:
        accuracy_chart = pd.DataFrame(
            data={"Training Accuracy": results['categorical_accuracy'],
                  "Epochs": list(range(1, epochs+1))
                  }
        )
        loss_chart = pd.DataFrame(
            data={"Training Loss": results['loss'],
                  "Epochs": list(range(1, epochs + 1))
                  }
        )


    st.write("## Accuracy and loss during training")
    st.write("### Accuracy")
    st.line_chart(accuracy_chart, x="Epochs")
    st.write("### Loss")
    st.line_chart(loss_chart, x="Epochs")

    st.write("-----")
    st.write("## Testing metrics ")
    test_data = read_patches(test_csv)[0]
    test_labels = read_patches(test_csv)[1]
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    predictions = model.predict(test_data)
    conf_matrix = confusion_matrix(test_labels.argmax(-1), predictions.argmax(-1))
    f1, precision, recall = compute_metrics(test_labels.argmax(-1), predictions.argmax(-1))

    st.write("### Confusion matrix")
    plot_confusion_matrix(conf_matrix, label_mapping)

    st.write("### ROC curve")
    auc_score = plot_roc_auc_curve(num_classes, test_labels, predictions, label_mapping)

    # Sample data
    data = {
        'Metric': ['Loss', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC'],
        'Value': [test_loss, test_acc, f1, precision, recall, auc_score]
    }
    st.write("### Classification Metrics")
    df_results = pd.DataFrame(data)
    df_results.set_index('Metric', inplace=True)
    st.table(df_results)


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
    data = data/255

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
            results = {'loss': [], 'categorical_accuracy': [],
                       'val_loss': [], 'val_categorical_accuracy': []}
            for epoch in range(epochs):
                train_csv = train_csv.sample(frac=1)
                counter=0
                acc_loss = 0
                acc_accuracy = 0
                for batch in range(0, train_csv.shape[0], batch_size):
                    df_batch = train_csv.iloc[batch:batch + batch_size]
                    patches, labels = read_patches(df_batch)
                    history = model.fit(patches, labels, verbose=0)
                    # print(f"{batch}: {history.history}")
                    counter+=1
                    acc_loss += history.history["loss"][0]
                    acc_accuracy += history.history["categorical_accuracy"][0]


                loss = acc_loss/counter
                accuracy = acc_accuracy/counter
                # print(f"\n END OF EPOCH: LOSS: {loss} - ACCURACY:{accuracy}")
                val_loss, val_accuracy = model.evaluate(val_data, val_labels)

                results['loss'].append(loss)
                results['val_loss'].append(val_loss)
                results['categorical_accuracy'].append(accuracy)
                results['val_categorical_accuracy'].append(val_accuracy)
                st.write(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}'
                         f' - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}')

            show_training_metrics(results, test_csv)

        else:
            results = {'loss': [], 'categorical_accuracy': []}
            for epoch in range(epochs):
                train_csv = train_csv.sample(frac=1)
                counter=0
                acc_loss = 0
                acc_accuracy = 0
                for batch in range(0, train_csv.shape[0], batch_size):
                    df_batch = train_csv.iloc[batch:batch + batch_size]
                    # print(df_batch)
                    patches, labels = read_patches(df_batch)
                    history = model.fit(patches, labels, verbose=0)
                    # print(f"{batch}: {history.history}")
                    counter += 1
                    acc_loss += history.history["loss"][0]
                    acc_accuracy += history.history["categorical_accuracy"][0]
                loss = history.history['loss'][0]
                accuracy = history.history['categorical_accuracy'][0]
                results['loss'].append(loss)
                results['categorical_accuracy'].append(accuracy)
                # print(f"Metrics in the epoch {epoch}: {history.history}")
                st.write(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')
            show_training_metrics(results, test_csv)
        time = time.time() - start_time
        time = time/60
        st.write(f"Training finished in {time:.2f} min")


st.set_page_config(
    page_title="Train your model",
    page_icon="images/favicon.png",
)
st.title("Train your model")
st.write("Please input any of the CSV files generated in the preprocessing section.")

st.write("Following this step, the option to configure the model, as well as "
         "crucial training settings like the optimizers and hyperparameters, will become available.")

st.write("## Load your data")
csv_file = st.file_uploader("Load the CSV file", type=['csv'])

if csv_file is not None:
    csv = load_csv(csv_file)

    # Show DataFrame
    st.write("Your CSV file:")
    st.dataframe(csv)

    label_mapping, num_classes = preprocessing_labels(csv)

    max_number_of_patches = st.slider("Select the maximum number of patches that can be randomly extracted "
                                      "from each image:", 1, 2000, step=1, value=100,
                                      help="If more patches are selected than the maximum allowed, "
                                           "all patches within the image will be used.")

    csv_with_keys = create_csv_of_random_keys(csv, max_number_of_patches)

    st.write("------")
    st.write("## Configure your model")

    choose_a_model = st.selectbox("Select a Transfer Learning model to perform your training",
                                  ("VGG16", "MobileNetV2", "ResNet50", "InceptionV3"),
                                  help="**ResNet** model demands the most computational resources, "
                                       "closely trailed by **InceptionV3**, **VGG16**, and the most resource-efficient "
                                       "of them all, **MobileNetV2**. Choose one taking this into consideration."
                                  )
    batch_size = st.slider("Batch Size", min_value=8, max_value=128, step=2, value=16,
                           help="Batch size refers to the quantity of data samples processed in a single forward "
                                "and backward pass by the model during training. This division of input data "
                                "into batches allows the model to update its internal weights after "
                                "handling each batch.")
    epochs = st.slider("Epochs", min_value=1, max_value=100, step=1, value=10,
                       help="Epoch refers to a complete pass through the entire training dataset during the training "
                            "phase of a model.")
    choose_optimizer = st.radio("Choose the type of Optimizer ", ("Adam", "SGD", "RMSProp", "Adagrad"),
                                help= "This algorithm is employed to adjust the neural network's weights "
                                      "throughout the training process. The choice of an optimizer "
                                      "shifts the optimization strategy employed by the neural network.")

    learning_rate = st.slider("Learning Rate", min_value=0.000001, max_value=0.1, step=0.000001, value=0.001,
                              format="%f",
                             help = "The learning rate serves as a critical hyperparameter within "
                                    "the optimization process of machine learning models. "
                                    "It dictates the magnitude of each step the optimization "
                                    "algorithm takes towards minimizing the loss function."
                                    "Choosing a **higher learning rate** speeds up convergence but can lead to "
                                    "overshooting and instability. Conversely, a **lower learning rate** promotes "
                                    "accuracy but may slow down convergence.")
    val = st.checkbox('Validation Set',
                      help="Validation data is a subset of the training data used to prevent overfitting. "
                           "It serves as an independent dataset for assessing the model's performance, "
                           "ensuring its effectiveness with unseen data. "
                           "Enable this button to monitor the model's performance with "
                           "validation metrics.")

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
        # data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),])

        if choose_a_model == "VGG16":
            base_model = keras.applications.VGG16(weights="imagenet", input_shape=patch_size,  include_top=False)
        elif choose_a_model == "MobileNetV2":
            base_model = keras.applications.MobileNetV2(weights="imagenet", input_shape=patch_size,  include_top=False)
        elif choose_a_model == "ResNet50":
            base_model = keras.applications.ResNet50(weights="imagenet", input_shape=patch_size,  include_top=False)
        elif choose_a_model == "InceptionV3":
            base_model = keras.applications.InceptionV3(weights="imagenet", input_shape=patch_size,  include_top=False)

        for layer in base_model.layers[:-2]:
            layer.trainable = False

        x = base_model.output
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.1)(x)
        output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs=base_model.input, outputs=output)
        model.compile(
            optimizer=choose_optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.categorical_accuracy],
        )

        train_csv_with_keys(csv_with_keys, batch_size, epochs)

        st.balloons()

        model_name = f"{choose_a_model}_{str(uuid.uuid4())}-" \
                     f"patch_size_{patch_size[0]}-" \
                     f"max_patch_num_{max_number_of_patches}-" \
                     f"batch_size_{batch_size}-" \
                     f"epoch_{epochs}-" \
                     f"optimizer_{choose_optimizer}" \
                     f".h5"
        current_directory = os.getcwd()
        model_path = os.path.join(current_directory, model_name)
        model.save(model_path)
        st.success(f'Model saved in saved in {model_path}')
        st.write("Copy the path of this trained model to **visualize** patches!")


