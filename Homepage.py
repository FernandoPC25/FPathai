import streamlit as st
st.set_page_config(
    page_title="Histopathology App",
    layout="wide",  #'wide' o 'centered'
    initial_sidebar_state="expanded"  # collapsed' o 'expanded'
)



st.write("""
# Histopathological prediction

*Brief Description*

Images obtained from [NIH Cancer Datasets](https://portal.gdc.cancer.gov/)
""")
