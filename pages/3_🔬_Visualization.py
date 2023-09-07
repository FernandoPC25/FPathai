import random

import cv2
import h5py
import keras
import matplotlib.cm as cm
import numpy as np 
import pandas as pd
import streamlit as st
import tensorflow as tf


def take_last_convolutional_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break  # Stop when you find the last convolutional layer
    last_conv_layer_name = last_conv_layer.name
    return last_conv_layer_name

def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    img = np.expand_dims(img, axis=0)
    model.layers[-1].activation = None
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.5):
    shape=(img.shape[0],img.shape[0])
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(shape)
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    img = np.asarray(img, np.int)
    jet_heatmap = np.asarray(jet_heatmap, np.int)
    blended_image = cv2.addWeighted(img, alpha, jet_heatmap, 1 - alpha, 0)
    return blended_image

def expand_image(image):
    image = np.expand_dims(image, axis=0)
    return image


# def show_patches_with_gradcam(patches, predictions, labels, num_rows=5, num_cols=5):
#     for i in range(num_rows):
#         cols = st.columns(num_cols)
#         for j in range(num_cols):
#             index = i * num_cols + j
#             if index < len(patches):
#                 heatmap = make_gradcam_heatmap(patches[index], loaded_model, last_conv_layer_name)
#                 patch_with_gradcam = save_and_display_gradcam(patches[index], heatmap)
#                 text=""
#                 for k, label in enumerate(labels):
#                     text += f" {label}: {round(((predictions[index][k])*100),3)}%"
#                 #text = f"Class 0:{predictions[index][0]} Class 1: {predictions[index][1]}"
#                 cols[j].image(patch_with_gradcam,
#                               caption=text, use_column_width=True, clamp=True, channels='RGB')


def show_patches_and_patches_with_gradcam(patches, predictions, labels, random_patches, num_rows=5):
    """
    Plots a number `num_rows` of images next to the Grad-CAM prediction superimposed on the image
    """
    col1, col2 = st.columns(2)
    col1.header("Original patch")
    col2.header("GradCAM prediction")
    for i in range(num_rows):
        heatmap = make_gradcam_heatmap(patches[i], loaded_model, last_conv_layer_name)
        patch_with_gradcam = save_and_display_gradcam(patches[i], heatmap)
        col1.image(patches[i],
                   caption=f"Patch {random_patches[i]}", use_column_width=True, channels="RGB")

        text = "|"
        for k, label in enumerate(labels):
            text += f" {label}: {round(((predictions[i][k]) * 100), 3)}% |"
        col2.image(patch_with_gradcam,
                   caption=text, use_column_width=True, clamp=True, channels='RGB')


def load_csv(csv_file):
    """
    Load the csv
    """
    df = pd.read_csv(csv_file)
    return df


st.set_page_config(
    page_title="Visualize",
    page_icon="images/favicon.png",
)
st.title("Visualize predictions")



st.write("In this application section, a visual perspective on the performance of the previously "
         "trained model can be obtained. The input that are required are:"
         "\n 1) The CSV that was used to train the data"
         "\n 2) The model created within the application"
         "\n 3) The patches created within the application"
         )


st.write("## 1) Introduce the original dataset")
st.write("Introduce the dataset that has been used to train your model. "
         "This step is necessary to know the possible labels that the problem may take.")
csv_file = st.file_uploader("Load the CSV file", type=['csv'])

if csv_file is not None:
    csv = load_csv(csv_file)
    st.write("Your CSV file:")
    st.dataframe(csv)
    h5_labels = np.array(csv["Label"].values)
    st.write(f"Your dataset has {len(np.unique(h5_labels))} labels: {np.unique(h5_labels)}")
    label_mapping = {label: idx for idx, label in enumerate(set(h5_labels))}
    labels=list(label_mapping)

    st.write("## 2) Enter a model created with the application")
    h5_model = st.text_input("Enter the path of the model.h5 trained with this dataset.",
                             help="This field should be something like: "
                                  "/path/to/directory/256-model-xx00xx00-xx00-0000-0xxx-xxxx0000xxxx000.h5")

    if h5_model.endswith(".h5"):
        try:
            loaded_model = tf.keras.models.load_model(h5_model)
            if isinstance(loaded_model, tf.keras.models.Model):
                #st.write(loaded_model.input)
                last_conv_layer_name = take_last_convolutional_layer(loaded_model)
                st.success(f"**Model** successfully loaded")
                # st.write(last_conv_layer_name)
                st.write("## 3) Enter the file path to the image patches to analyze.")
                patch_path = st.text_input("Enter the path of the created patches")
                if patch_path.endswith(".h5"):
                    try:
                        patches = []
                        with h5py.File(patch_path, 'r') as data:
                            number_keys = list(data.keys())
                            random_patches = random.sample(number_keys, 25)
                            for key in random_patches:
                                patch = data[key][:]
                                patches.append(patch)
                        patches = np.array(patches)
                        predictions = loaded_model.predict(patches)
                        st.success(f"**Patches** successfully loaded")
                        with st.form("visualize"):
                            submitted = st.form_submit_button("Visualize random patches!")
                            st.write("Visualize random patches alongside the corresponding "
                                     "predictions from the loaded model.")
                            st.write("-----")
                            if submitted:
                                show_patches_and_patches_with_gradcam(patches, predictions, labels, random_patches)
                    except:
                        st.warning("Introduce a valid patches path created within the application."
                                   "\n\n Remember to add a patch file with the same dimension of the "
                                   "uploaded model.", icon="⚠️")
                elif not patch_path:
                    st.info("Please specify the directory containing the .h5 **patches**", icon="ℹ")
                else:
                    st.error("Introduce a correct format for the patches", icon="🚨")
            else:
                st.error("Introduce a correct format for the model", icon="🚨")
        except:
            st.warning("Introduce a valid model created within the application", icon="⚠️")
    elif not h5_model:
        st.info("Please specify the directory containing the .h5 model", icon="ℹ")
    else:
        st.error("Introduce a correct format for the model", icon="🚨")

