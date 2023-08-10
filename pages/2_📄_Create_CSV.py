import os

import h5py
import numpy as np
import pandas as pd
import streamlit as st


def generate_csv_with_h5_images():
    """
    Generate the csv
    """
    directory_control = st.text_input("Path to the control folder:")
    if not directory_control:
        st.info("Introduce the path of the control subjects.", icon="ℹ")
        return
    if not os.path.isdir(directory_control):
        st.error("Invalid directory path.", icon="🚨")
        return

    h5_images_control = [filename for filename in os.listdir(directory_control) if filename.lower().endswith(".h5")]

    if not h5_images_control:
        st.info("No H5 images found in the directory.")
        return
    st.success(f"Found {len(h5_images_control)} H5 image(s) in the directory.")

    ########

    directory_tumor = st.text_input("Path to the tumor folder:")
    if not directory_tumor:
        st.info("Introduce the path of the tumor subjects.")
        return
    if not os.path.isdir(directory_tumor):
        st.error("Invalid directory path.")
        return

    if directory_tumor == directory_control:
        st.error("The paths are the same! Please change the directory.")
        return

    h5_images_tumor = [filename for filename in os.listdir(directory_tumor) if filename.lower().endswith(".h5")]

    if not h5_images_tumor:
        st.info("No H5 images found in the directory.")
        return

    st.success(f"Found {len(h5_images_tumor)} H5 image(s) in the directory.")

    csv_filename = st.text_input("Enter the CSV filename (without extension):")

    if not csv_filename:
        st.info("Enter a CSV filename.")
        return

    csv_filename = csv_filename + ".csv"

    csv_file = generate_csv_from_h5_files(directory_control, directory_tumor, csv_filename)

    st.success(f"CSV file '{csv_filename}' with **{len(h5_images_control)} control image(s)** and "
               f"**{len(h5_images_tumor)} tumor image(s)** created successfully."
               f"\n\n Now you can download it.")

    st.download_button(label="Download CSV", data=open(csv_file, 'rb').read(), file_name=csv_filename,
                       mime='text/csv')


def generate_csv_from_h5_files(folder_path_control, folder_path_tumor, csv_filename):
    csv_data = []
    for filename in os.listdir(folder_path_control):
        if filename.endswith('.h5'):
            full_path = os.path.join(folder_path_control, filename)
            patient_id = filename.split('/')[-1].split('-')[:3]
            patient_id = '-'.join(patient_id)
            file_name = os.path.splitext(filename)[0]
            file_name = file_name + '.h5'
            csv_data.append({'Path': full_path, 'WSI_name': file_name, 'Patient_ID': patient_id, 'Label': "Control"})

    for filename in os.listdir(folder_path_tumor):
        if filename.endswith('.h5'):
            full_path = os.path.join(folder_path_tumor, filename)
            patient_id = filename.split('/')[-1].split('-')[:3]
            patient_id = '-'.join(patient_id)
            file_name = os.path.splitext(filename)[0]
            file_name = file_name + '.h5'
            csv_data.append({'Path': full_path, 'WSI_name': file_name, 'Patient_ID': patient_id, 'Label': "Tumor"})

    df = pd.DataFrame(csv_data)
    st.write("Check the CSV that you are about to download:")
    st.dataframe(df, use_container_width=True)
    csv_filename = csv_filename
    df.to_csv(csv_filename, index=False)

    return csv_filename


def get_patches_from_h5_folder(folder, label):
    h5_files = [file for file in os.listdir(folder) if file.endswith('.h5')]
    total_iterations = len(h5_files)
    progress_bar = st.progress(0)
    patches = []
    labels = []
    patches_count = 0
    for (i, h5_file) in enumerate(h5_files):
        h5_image_path = os.path.join(folder, h5_file)
        with h5py.File(h5_image_path, 'r') as data:
            number_keys = list(data.keys())
            patches_count += len(number_keys)
            for key in number_keys:
                patch = data[key][:]
                patches.append(patch)
        progress = (i + 1) / total_iterations
        st.write(f"Found {len(number_keys)} patches for the image {h5_file}")
        progress_bar.progress(progress, text="Reading patches... Please wait.")
    if label == "tumor":
        labels.append([0] * patches_count)
    if label == "control":
        labels.append([1] * patches_count)

    labels = np.eye(2)[labels]
    labels = labels.reshape(labels.shape[1], labels.shape[2])

    return patches, labels



def create_dataset_tumor(directory_tumor):
    if not directory_tumor:
        st.info("Introduce the path of the tumor subjects.")
        return None
    if not os.path.isdir(directory_tumor):
        st.error("Invalid directory path.")
        return None

    h5_images_control = [filename for filename in os.listdir(directory_tumor) if filename.lower().endswith(".h5")]

    if not h5_images_control:
        st.warning("No H5 images found in the directory.", icon="⚠️")
        return None
    else:
        st.success(f"Found {len(h5_images_control)} H5 image(s) in the tumor directory.")
        patches, labels = get_patches_from_h5_folder(directory_tumor, "tumor")
        return patches, labels


def create_dataset_control(directory_control):
    if not directory_control:
        st.info("Introduce the path of the control subjects.")
        return None
    if not os.path.isdir(directory_control):
        st.error("Invalid directory path.")
        return None

    h5_images_control = [filename for filename in os.listdir(directory_control) if filename.lower().endswith(".h5")]

    if not h5_images_control:
        st.warning("No H5 images found in the directory.", icon="⚠️")
        return None
    else:
        st.success(f"Found {len(h5_images_control)} H5 image(s) in the control directory.")
        patches, labels = get_patches_from_h5_folder(directory_control, "control")
        return patches, labels


st.set_page_config(
    page_title="Create CSV",
)
st.title("Create CSV")
st.write("The CSV is what the model needs in order to perform the training")

generate_csv_with_h5_images()
