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

import streamlit as st
import graphviz

# Create a graphlib graph object
#
#
# st.graphviz_chart('''
#     digraph {
#         "Create Patches" -> "Create CSV"
#         "Otra cosa" -> "Train Model"
#         "Create CSV" -> "Train Model"
#
#     }
# ''')
#
# st.graphviz_chart('''
#     digraph {
#         "A" -> "B"
#         "A" -> "D"
#         "A" -> "E"
#         "B" -> "H"
#         "B" -> "C"
#         "C" -> "F"
#         "D" -> "G"
#         "E" -> "G"
#         "E" -> "H"
#
#     }
# ''')
