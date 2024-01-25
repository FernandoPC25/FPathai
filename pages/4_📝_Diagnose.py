OPENSLIDE_PATH = r'c:\Users\Fernando\Desktop\FPC\MASTER UGR\4-TFM\openslide-win64\bin'
import os

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import OpenSlide
import pandas as pd
import numpy as np
import os
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.exposure.exposure import is_low_contrast
from scipy.ndimage import binary_dilation, binary_erosion
import h5py

import streamlit as st
import random
import matplotlib.pyplot as plt
import tensorflow as tf


def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask


def get_mask(slide, level='max', RGB_min=50):
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    img_RGB = np.transpose(np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level


def get_patches_and_coordinates(slide_path, patch_size, max_patches_per_slide=5000):
    try:
        slide = OpenSlide(slide_path)
    except:
        return

    mask, mask_level = get_mask(slide)
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=3)

    mask_level = len(slide.level_dimensions) - 1

    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    try:
        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
        patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
        i = 0

        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                   range(0, ymax, patch_size_resized[0])]

        patches = []
        indxy = []
        for x, y in indices:
            x_mask = int(x / ratio_x)
            y_mask = int(y / ratio_y)

            if mask[x_mask, y_mask] == 1:
                patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                try:
                    mask_patch = get_mask_image(np.array(patch))
                    mask_patch = binary_dilation(mask_patch, iterations=3)
                except Exception as e:
                    print(e)
                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    if resize_factor != 1.0:
                        patch = patch.resize(patch_size)
                    patch = np.array(patch)
                    indxy.append((x_mask, y_mask))
                    patches.append(patch)
                    i += 1

            if i >= max_patches_per_slide:
                break

        if i == 0:
            print("no patch extracted for slide {}".format(slide_path))

    except Exception as e:
        print("error with slide id {} patch {}".format(slide_path, i))
        print(e)

    patches = np.array(patches)

    return indxy, patches

def predict_majority_class(predictions):
    predicted_class = np.argmax(np.bincount(np.argmax(predictions, axis=1)))
    print("ke:", np.bincount(np.argmax(predictions, axis=1)))
    print("predicted_class: ", predicted_class)
    return predicted_class


def detect_correct_and_wrong_coordinates(coords, predicted_class, threshold=0.5):
    x, y = zip(*coords)
    correct_coords = []
    wrong_coords = []
    for i in range(len(predictions[:, predicted_class])):
        if predictions[:, predicted_class][i] >= threshold:
            correct_coords.append((x[i], y[i]))
        else:
            wrong_coords.append((x[i], y[i]))
    return correct_coords, wrong_coords

def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocessing_labels(csv):
    h5_labels = np.array(csv["Label"].values)
    label_mapping = {label: idx for idx, label in enumerate(set(h5_labels))}
    num_classes = len(label_mapping)
    return label_mapping, num_classes


def plot_figure_and_patches(svs_image_path, label_mapping, predicted_class, patch_size, coords, threshold):
    # Plot the original image
    svs_image = openslide.open_slide(svs_image_path)
    level = len(svs_image.level_dimensions) - 1
    svs_imagen_read = svs_image.read_region((0, 0), level, svs_image.level_dimensions[level])
    figure = np.array(svs_imagen_read)
    ratio = svs_image.level_dimensions[0][0] / svs_image.level_dimensions[level][0]
    plt.figure(figsize=(25, 25))
    plt.imshow(figure)


    # Plot the predicted patches
    correct_coords, wrong_coords = \
        detect_correct_and_wrong_coordinates(coords, predicted_class=predicted_class, threshold=threshold)

    plt.title(
        f"Classified as {list(label_mapping.keys())[predicted_class]} with {round(100 * (len(correct_coords) / len(predictions)), 3)}%")
    if len(correct_coords) > 0:
        x1, y1 = zip(*correct_coords)
        plt.scatter(x1, y1, s=round(4 * patch_size[0] / ratio), color="green", alpha=0.3, marker='s',
                    label=f'{len(correct_coords)} predicted as {list(label_mapping.keys())[predicted_class]}')
    if len(wrong_coords) > 0:
        x2, y2 = zip(*wrong_coords)
        plt.scatter(x2, y2, s=round(4 * patch_size[0] / ratio), color="red", alpha=0.4, marker='s',
                    label=f'{len(wrong_coords)} not predicted as {list(label_mapping.keys())[predicted_class]}')
    plt.legend()
    st.pyplot(plt)


st.set_page_config(
    page_title="Predict image",
    page_icon="images/favicon.png",
)
st.title("Diagnose")
st.write("In this application section, patch analysis is employed, and a majority voting mechanism "
         "is utilized for diagnosing images in .svs format. The process is facilitated by a pre-trained "
         "model previously obtained with **FPathai**.")

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
    print("label_mapping: ", label_mapping)
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
                patch_size = loaded_model.layers[0].input_shape[0][1:3]
                st.success(f"**Model** successfully loaded")
                # st.write(last_conv_layer_name)
                st.write("## 3) Enter the file path to the svs image to analyze.")
                svs_image_path = st.text_input("Enter the path to the svs image:",
                                               help="Analyze the image using patches.")
                if os.path.exists(svs_image_path):
                    if svs_image_path.endswith(".svs"):
                        slide = OpenSlide(svs_image_path)
                        thumbnail = slide.get_thumbnail((1024, 1024))
                        with st.form("diagnose"):
                            st.image(thumbnail, caption="Image to diagnose", use_column_width=True)
                            max_patches = st.slider('Choose the maximum number of patches for the image',
                                                    min_value=1, max_value=10000, value=1000, step=1,
                                                    help="The larger this value is, the more patches will be created.")
                            threshold = st.slider('Select a threshold',
                                                  min_value=0.0, max_value=1.0, step=0.001, value=0.5, format="%f",
                                                  help="When making predictions, the usual practice is to select "
                                                       "the value with the highest score. In this scenario,"
                                                       " we provide the user with the option to choose a threshold "
                                                       "value for classifying the patch.")

                            submitted = st.form_submit_button("Diagnose this image!")
                            if submitted:
                                with st.spinner("Generating patches..."):
                                    coords, patches = get_patches_and_coordinates(svs_image_path,
                                                                                  patch_size=patch_size,
                                                                                  max_patches_per_slide=max_patches)
                                st.success(f"{patches.shape[0]} patches with {patch_size} size has been generated!")
                                with st.spinner(f"Evaluating patches..."):
                                    predictions = loaded_model.predict(patches)
                                    print(predictions)

                                    predicted_class = predict_majority_class(predictions)
                                    st.info(f"The image has been classified as "
                                            f"**{list(label_mapping.keys())[predicted_class]}**."
                                            f"\n The diagnose can be found here:", icon="👨‍⚕️")
                                plot_figure_and_patches(svs_image_path, label_mapping, predicted_class,
                                                            patch_size, coords, threshold)

                elif not svs_image_path:
                    st.info("Please specify the directory containing the .svs format images for which you wish to generate "
                            "patches.", icon="ℹ")
                else:
                    st.error("Introduce a correct format for the image", icon="🚨")
            else:
                st.error("Introduce a correct format for the model", icon="🚨")
        except:
            st.warning("Introduce a valid model created within the application", icon="⚠️")
    elif not h5_model:
        st.info("Please specify the directory containing the .h5 model", icon="ℹ")
    else:
        st.error("Introduce a correct format for the model", icon="🚨")


# if csv_file is not None:
#     csv = load_csv(csv_file)
#     st.write("Your CSV file:")
#     st.dataframe(csv)
#     h5_labels = np.array(csv["Label"].values)
#     st.write(f"Your dataset has {len(np.unique(h5_labels))} labels: {np.unique(h5_labels)}")
#     label_mapping = {label: idx for idx, label in enumerate(set(h5_labels))}
#     print(label_mapping)
#     labels=list(label_mapping)
#
#     svs_image_path = st.text_input("Enter the path to the svs image:",
#                                    help="blalbal")
#
#     if os.path.exists(svs_image_path):
#         if svs_image_path.endswith(".svs"):
#             slide = OpenSlide(svs_image_path)
#             thumbnail = slide.get_thumbnail((1024, 1024))
#             st.image(thumbnail, caption="Image to diagnose", use_column_width=True)
#
#             h5_model = st.text_input("Enter the path of the model.h5 trained with this dataset.",
#                                      help="This field should be something like: "
#                                           "/path/to/directory/256-model-xx00xx00-xx00-0000-0xxx-xxxx0000xxxx000.h5")
#             if h5_model.endswith(".h5"):
#                 try:
#                     loaded_model = tf.keras.models.load_model(h5_model)
#                     if isinstance(loaded_model, tf.keras.models.Model):
#                         patch_size = loaded_model.layers[0].input_shape[0][1:3]
#                         with st.spinner(f"Generating and evaluating patches of {patch_size}..."):
#                             coords, patches = get_patches_and_coordinates(svs_image_path, patch_size)
#
#                             predictions = loaded_model.predict(patches)
#
#                             predicted_class = predict_majority_class(predictions)
#
#                             plot_figure_and_patches(svs_image_path, label_mapping, predicted_class, patch_size, coords)
#
#                         st.success("Done!")
#
#
#
#                     else:
#                         st.error("Introduce a correct format for the model", icon="🚨")
#                 except:
#                     st.warning("Introduce a valid model created within the application", icon="⚠️")
#
#
#             elif not h5_model:
#                 st.info("Please specify the directory containing the .h5 model", icon="ℹ")
#             else:
#                 st.warning("Introduce a model in .h5 format", icon="⚠️")
#
#
#         else:
#             st.warning("Please introduce an image in .svs format", icon= "⚠️")
#
#     elif not svs_image_path:
#         st.info("Please specify the directory containing the .svs format images for which you wish to generate "
#                 "patches.", icon="ℹ")
#     else:
#         st.error("Invalid directory path.", icon="🚨")


    #C:/Users/Fernando/Desktop/MASTER UGR/4-TFM/TFM-streamlit/DIAGNOSE/TCGA-22-4595-11A-01-BS1.460293a1-334c-4a57-999d-a9ba82fb289b.svs

    # C:/Users/Fernando/Desktop/MASTER UGR/4-TFM/TFM-streamlit/project/VGG16_b8820d06-c221-4848-96c1-059e4c1f59b5-patch_size_256-max_patch_num_50-batch_size_32-epoch_3-optimizer_Adam.h5