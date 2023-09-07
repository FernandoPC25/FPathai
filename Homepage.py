import streamlit as st
from PIL import Image
st.set_page_config(
    page_title="Histopathological prediction",
    page_icon="images/favicon.png",
    layout="centered",  #'wide' o 'centered'
    initial_sidebar_state="expanded"  # collapsed' o 'expanded'
)

# st.sidebar.image("favicon.png", width= 50)

image = Image.open('images/logo.png')

st.image(image, width=300, use_column_width="always")


st.write("""
# Histopathological prediction

This application enables users to perform comprehensive image analysis on WSI, specifically, svs. 
It offers the capability to segment large WSI images into smaller patches for enhanced processing efficiency. 
Users can then harness these patches to train machine learning models, 
enabling advanced analysis and pattern recognition. 
Furthermore, the application provides a user-friendly interface for visualizing the model's predictions 
on these patches.

SVS images can be obtained from [NIH Cancer Datasets](https://portal.gdc.cancer.gov/)
""")

st.write("When images in the SVS format are available on the user's device, "
         "they must be reorganized as follows to ensure compatibility with the application:")

st.write(
    "**Step 1: Create a Main Folder**"
    "\n\nBegin by creating a main folder on your local machine or storage device. "
    "This main folder will serve as the central hub for all your experiment data. "
    "Give it a descriptive name that relates to your project."
    
    "\n\n**Step 2: Class Subfolders** "
    "\n\n Within the main folder, create subfolders to represent each distinct class or "
    "category of your experiment. For example, your classes might be 'Control', 'Tumor', and 'Metastasis'."
    
    "\n\n**Step 3: Organize with Subfolders** "
    "\n\nThe subfolders will hold the SVS images that correspond to their respective classes."
    "For doing so, simply place the SVS (Whole Slide Images) into the appropriate sub-subfolders according to their "
    "respective classes. This structured organization ensures that your data remains clear and accessible for analysis."
)

st.write("\n\n\n")

image1 = Image.open('images/methodology.png')

st.image(image1, caption='Prepare the data for the application', clamp=True)

st.write("Once the images have been organized in this manner, the application can be utilized seamlessly,"
         " **eliminating the necessity for any coding involvement!**")

image2 = Image.open('images/FPathai-diagram.png')

st.image(image2, caption='Workflow of the application', clamp=True)

