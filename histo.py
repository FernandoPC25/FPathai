import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
from io import BytesIO
import base64

from PIL import Image
# from weasyprint import HTML
import csv
import time


from utils import generate_csv_from_h5_files, generate_csv_with_h5_images


############
# Introduccion
############

st.write("""
# Histopathological prediction

*Brief Description*

Images obtained from [NIH Cancer Datasets](https://portal.gdc.cancer.gov/)
""")

############
# Añadir CSV y parámetros del modelo
############

st.write("------")
uploaded_files = st.file_uploader("Choose the CSV with the all the required information to "
                                  "perform the experiment:", accept_multiple_files=False)

with st.sidebar:
    decision = st.radio(
        "Do you need to create a CSV?",
        ('Yes', 'No'))
    if decision == 'Yes':
        st.title("Create CSV")
        generate_csv_with_h5_images()
    else:
        st.write("Then upload the CSV to start the training.")



st.write("------")
st.title("Train your model")







############
# Preprocesamiento
############


############
# Entrenamiento
############


############
# Testing
############


# ############
# # Generate PDF
# ############
#
# import streamlit as st
# from fpdf import FPDF
# import base64
# from io import BytesIO
# from xhtml2pdf import pisa
#
#
# # export_as_pdf = st.button("Export PDF")
#
#
# def create_download_link(val, filename):
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
#
#
# if export_as_pdf:
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(40, 10, "Prueba de la generación del PDF")
#
#     html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
#
#     st.markdown(html, unsafe_allow_html=True)
#
#
# st.write("------- \n # Resto")
#
# a = st.button('Hit me')
#
# st.checkbox('Check me out')
#
# #st.download_button('On the dl', data)
# #st.checkbox('Check me out')
# st.radio('Radio', [1,2,3])
# st.selectbox('Select', [1,2,3])
# st.multiselect('Multiselect', [1,2,3])
# st.slider('Slide me', min_value=0, max_value=10)
# st.select_slider('Slide to select', options=[1,'2'])
# st.text_input('Enter some text')
# st.number_input('Enter a number')
# st.text_area('Area for textual entry')
# st.date_input('Date input')
# st.time_input('Time entry')
# st.file_uploader('File uploader')
# st.camera_input("一二三,茄子!")
# st.color_picker('Pick a color')
#
#
# folder_path = "C:/Users/Fernando/Desktop/MASTER UGR/VisionComputador/Practica1/Practica1/"  # Reemplaza esto con la ruta a la carpeta que desees explorar
# files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#
# selected_files = st.multiselect("Seleccionar archivos:", files_in_folder)
#
# for selected_file in selected_files:
#     file_path = os.path.join(folder_path, selected_file)
#     with open(file_path, "r") as file:
#         file_contents = file.read()
#     st.text(f"Contenido del archivo {selected_file}:")
#     st.code(file_contents)