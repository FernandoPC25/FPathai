import os
import csv
import streamlit as st
import pandas as pd

def generate_csv_with_h5_images():
    """
    Generate the csv
    """
    directory_control = st.text_input("Path to the control folder:")
    if not directory_control:
        st.info("Introduce the path of the control subjects.")
        return
    if not os.path.isdir(directory_control):
        st.error("Invalid directory path.")
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