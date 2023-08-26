OPENSLIDE_PATH = r'c:\Users\Fernando\Desktop\MASTER UGR\4-TFM\openslide-win64\bin'
import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import os

import h5py
import numpy as np
import streamlit as st
from openslide import OpenSlide
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.color import rgb2hsv
from skimage.exposure.exposure import is_low_contrast
from skimage.filters import threshold_otsu

import os
import h5py
import pandas as pd
import pandas as pd
import numpy as np
from openslide import OpenSlide
from multiprocessing import Pool, Value, Lock
import os
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion
import argparse
import logging
import pickle
import h5py
from itertools import tee
import matplotlib.pyplot as plt

import streamlit as st
import time





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
    # read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level


def extract_patches(slide_path, mask_path, patch_size, patches_output_dir, slide_id, max_patches_per_slide=2000):
    # patch_folder = os.path.join(patches_output_dir, slide_id)
    patch_folder = patches_output_dir  # TODO CAMBIAR ESTO!!!!!!!!!!!!!
    print("patch_folder: ", patch_folder)
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)
    # else:
    #     return
    try:
        slide = OpenSlide(slide_path)
    except:
        return

    # patch_folder_mask = os.path.join(mask_path, slide_id)

    patch_folder_mask = mask_path  # TODO CAMBIAR ESTO!!!!!!!!!!!

    mask_path = os.path.join(patch_folder_mask, slide_id)
    if not os.path.exists(mask_path + "_mask.npy"):
        # print(os.path.join(patch_folder_mask, slide_id, "_mask.npy"))
        # os.makedirs(patch_folder_mask)
        mask, mask_level = get_mask(slide)
        mask = binary_dilation(mask, iterations=3)
        mask = binary_erosion(mask, iterations=3)
        np.save((mask_path + "_mask.npy"), mask)
        # print("path mask", os.path.join(mask_path, slide_id, '_mask.npy'))
    else:
        mask = np.load(mask_path + "_mask.npy")

    mask_level = len(slide.level_dimensions) - 1

    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    try:
        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        print("ratio_x: ", ratio_x)
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]
        print("ratio_y: ", ratio_y)

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

        # handle slides with 40 magnification at base level
        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
        print("resize_factor: ", resize_factor)
        # resize_factor = resize_factor * args.dezoom_factor
        print("patch_size[0]: ", patch_size[0])
        patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
        print("patch_size_resized: ", patch_size_resized)
        i = 0

        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                   range(0, ymax, patch_size_resized[0])]
        # print("indices: ", indices)
        # np.random.seed(5)
        # np.random.shuffle(indices)
        path_h5 = os.path.join(patch_folder, slide_id + '.h5')
        # path_h5 = os.path.join(slide_id + '.h5')
        print("path_h5: ", path_h5)
        if os.path.exists(path_h5):
            print('Image already converted')
            return len(list(h5py.File(path_h5, 'r')))
        with h5py.File(path_h5, 'w') as f:
            for x, y in indices:
                # check if in background mask
                x_mask = int(x / ratio_x)
                # print("x_mask: ", x_mask)
                y_mask = int(y / ratio_y)
                # print("y_mask: ", y_mask)
                # print("mask[x_mask, y_mask]: ", mask[x_mask, y_mask])
                if mask[x_mask, y_mask] == 1:
                    patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                    # print("patch: ", patch)
                    try:
                        mask_patch = get_mask_image(np.array(patch))
                        mask_patch = binary_dilation(mask_patch, iterations=3)
                    except Exception as e:
                        print("PRIMERA EXCEPCION: error with slide id {} patch {}".format(slide_id, i))
                        print(e)
                    if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                        if resize_factor != 1.0:
                            patch = patch.resize(patch_size)
                        img_idx = str(i)
                        patch = np.array(patch)
                        dset = f.create_dataset(img_idx, data=patch)
                        i += 1

                if i >= max_patches_per_slide:
                    break

        if i == 0:
            print("no patch extracted for slide {}".format(slide_id))

    except Exception as e:
        print("error with slide id {} patch {}".format(slide_id, i))
        print(e)

    return i


def get_slide_id(slide_name):
    return slide_name.split('.')[0] + '.' + slide_name.split('.')[1]



def count_svs_files_in_immediate_subfolders(folder_path):
    total_svs_count = 0
    svs_counts = {}
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        svs_count = sum(1 for file in os.listdir(subfolder_path) if file.endswith('.svs'))
        if svs_count > 0:
            svs_counts[subfolder] = svs_count
            total_svs_count += svs_count
    return svs_counts, total_svs_count


def create_patches_and_csv(root_folder, patch, total_svs_count):
    patch_size = (patch, patch)
    csv_data = []

    h5_data_path = os.path.join(root_folder, "h5_data")
    if not os.path.exists(h5_data_path):
        os.makedirs(h5_data_path)

    mask_path = os.path.join(h5_data_path, "mask")
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    patch_path = os.path.join(h5_data_path, f"patches{patch}")
    if not os.path.exists(patch_path):
        os.makedirs(patch_path)
    # We go through all the folders
    with st.spinner("Preprocessing your data. Please wait."):
        progress_bar = st.progress(0)
        k=0
        for root, subfolder, files in os.walk(root_folder):
            if "h5_data" in subfolder:
                subfolder.remove("h5_data")
            for file in files:
                # if k!=total_svs_count:
                if file.endswith(".svs"):
                    # svs files must be in subfolders
                    if root != root_folder:
                        # Get the absolute path of the .svs image
                        svs_path = os.path.join(root, file)

                        # slide_id
                        slide_id = get_slide_id(file)

                        # Create the patches (using svs_path)
                        progress = (k + 1) / total_svs_count
                        progress_bar.progress(progress, text=f"Creating patches of {slide_id}...")
                        number_patches = extract_patches(svs_path, mask_path, patch_size, patch_path, slide_id)

                        # Create the name of the .h5 file
                        h5_filename = file[:-4] + ".h5"

                        # Get the Patient ID
                        patient_id = '-'.join(file.split('-')[:3])

                        # Create the absolute paths for the .h5 images
                        h5_full_path = os.path.join(patch_path, h5_filename)

                        # Get the label (subfolder name)
                        label = os.path.basename(root)

                        # Add to CSV
                        csv_data.append([h5_full_path, h5_filename, patient_id, label, number_patches])
                        k+=1
                # else:
                #     st.success(f"Patches created in {patch_path}")
            progress_bar.empty()
        st.info(f"Patches created in {patch_path}", icon="ℹ️")

    # Create dataframe
    df = pd.DataFrame(csv_data, columns=["Path", "WSI_name", "Patient_ID", "Label", "num_patches"])

    # Create CSV in the root folder
    csv_file_path = os.path.join(root_folder, f"h5_data_{patch}x{patch}.csv")
    df.to_csv(csv_file_path, index=False)
    st.info(f"CSV prepared and created in {csv_file_path}:", icon="ℹ️")
    return df




st.set_page_config(
    page_title="Create patches",
)
st.title("Create patches and CSV")
st.write("In this section the preprocessing of your data images will be performed."
         "\n 1) The first step is to choose a main folder path, so we can "
         "scan its subfolders to identify and extract .svs image files."
         "\n 2) After establishing the main folder path, the next phase involves the user's precise "
         "selection of a patch size. Subsequently, this carefully chosen patch size is utilized to "
         "systematically extract patches from the identified .svs images, which are then"
         " stored in a compressed .h5 format. "
         "Upon completion, a CSV file is generated, encapsulating the information required for the model.")

st.write("## 1) Enter the path")
root_folder = st.text_input("Enter the path to the main folder:")

#if st.button("Count .svs Files"):
if os.path.exists(root_folder):
    svs_counts, total_svs_count = count_svs_files_in_immediate_subfolders(root_folder)
    # Avoid 1 label
    if len(svs_counts) == 1:
        st.warning("There must be more that one folder containing svs images.", icon= "⚠️")
    elif svs_counts:
        st.success(f"There are a total of {total_svs_count} .svs images:")
        for folder, count in svs_counts.items():
            st.write(f"- {folder}: {count} .svs files")

        st.write("----")
        st.write("## 2) Select the patch size")
        with st.form("my_form"):
            st.write("Select a patch size")
            patch = st.slider('Choose the patch for your images', min_value=2, max_value=1024, value=256, step=2,
                              help="The larger this value is, the larger the patch will be created.")

            submitted = st.form_submit_button("Create patches")
            if submitted:
                #st.write(f"Your dataset with patch size of {patch, patch}")
                patch_size = (patch, patch)
                df = create_patches_and_csv(root_folder, patch, total_svs_count)

        finished = False
        if submitted:
            finished = True

        if finished:
            st.success(f"Now you can train your model with the created CSV:", icon="✅")
            st.dataframe(df, use_container_width=True)

    else:
        st.warning("No .svs files found in immediate subfolders.", icon= "⚠️")
elif not root_folder:
    st.info("Please specify the directory containing the .svs format images for which you wish to generate "
            "patches.", icon="ℹ")
else:
        st.error("Invalid directory path.", icon= "🚨")




# def svs_images():
#     """
#     Generate the csv
#     """
#     root_folder = st.text_input("Path to the folder with the folders containing the svs images:")
#     if not root_folder:
#         st.info("Please specify the directory containing the .svs format images for which you wish to generate patches.", icon="ℹ")
#         return
#     if not os.path.isdir(root_folder):
#         st.error("Invalid directory path.", icon="🚨")
#         return
#
#     # svs_images = [filename for filename in os.listdir(directory) if filename.lower().endswith(".svs")]
#
#     svs_with_folders = []
#
#     for folder_name, subfolders, filenames in os.walk(root_folder):
#         if "h5_data" in subfolders:
#             subfolders.remove("h5_data")  # Exclude "h5_data" folder from further traversal
#         for filename in filenames:
#             if filename.endswith(".svs"):
#                 parent_folder_name = os.path.basename(folder_name)
#                 svs_with_folders.append((os.path.join(folder_name, filename), parent_folder_name))
#
#     for svs_name, folder_name in svs_with_folders:
#         st.write("SVS File:", svs_name)
#         # print("From Folder:", folder_name)
#         print("-" * 40)
#
#     if not svs_images:
#         st.info("No .svs images found in the directory.")
#         return
#
#     st.success(f"Found {len(svs_with_folders)} .svs image(s) in the directory.")
#
#     for svs_image, folder_name in svs_with_folders:
#         st.write(folder_name)
#         full_path = os.path.join(root_folder, svs_image)
#         #openslide.open_slide(full_path).get_thumbnail(size=(1024, 1024))
#         st.image(openslide.open_slide(full_path).get_thumbnail(size=(1024, 1024)), caption=f"{svs_image}")
#
#     mask_path = "mask"
#     patch_path = "patches"
#     mask_path = os.path.join(root_folder, mask_path)
#     patch_path = os.path.join(root_folder, patch_path)
#     st.success(f"The patches for these images are going to be created in {patch_path}")
#
#     with st.form("my_form"):
#         st.write("Select a patch size")
#         patch = st.slider('Choose the patch for your images', min_value=2, max_value=1024, value=256, step=2,
#                           help="The larger this value is, the larger the patch will be created.")
#
#         submitted = st.form_submit_button("Create patches")
#         if submitted:
#             st.write(f"You have selected a patch size of {patch,patch}")
#             patch_size = (patch, patch)
#
#             progress_bar = st.progress(0)
#             total_images = len(svs_images)
#             for (i, s) in enumerate(svs_images):
#                 slide_path = os.path.join(root_folder, svs_images[i])
#                 slide_id = get_slide_id(svs_images[i])
#                 progress = (i + 1) / total_images
#                 progress_bar.progress(progress, text=f"Creating patches of {slide_id}... Please wait")
#                 extract_patches(slide_path, mask_path, patch_size, patch_path, slide_id)
#             st.success("Patches created")
#
#
# svs_images()
