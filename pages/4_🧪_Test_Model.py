import streamlit as st
import time
st.set_page_config(
    page_title="Test your model",
)
st.title("Test your model")

import streamlit as st
import pandas as pd
import numpy as np
#
# chart_data = pd.DataFrame(
#     data={"Accuracy": [0.6, 0.62, 0.43, 0.61, 0.70, 0.95],
#           "Val_Accuracy": [0.2, 0.4, 0.23, 0.31, 0.10, 0.15],
#           "Epochs": [1,2,3,4,5,6]},
#     )
#
# st.line_chart(chart_data, y=["Accuracy", "Val_Accuracy"] , x="Epochs",
#               use_container_width=True)

st.write("Inputs: The model & the test image"
         "\n\nOutputs: Test image, divided by patches and each patch with a probability. Final classification")

st.write("Hay que hacer una división en patches de la imagen de test (igual que las de entrenamiento)."
         "Después, clasificar esos patches y recolocarlos de tal forma que parezca la imagen original")

import pandas as pd
import streamlit as st

data_df = pd.DataFrame(
    {
        "sales": [200, 550, 1000, 80],
    }
)

st.data_editor(
    data_df,
    column_config={
        "sales": st.column_config.ProgressColumn(
            "Sales volume",
            help="The sales volume in USD",
            format="$%f",
            min_value=0,
            max_value=1000,
        ),
    },
    hide_index=True,
)

# import streamlit as st
# import pandas as pd
# import numpy as np
#
# # Generate example data
# data = {
#     'x': np.arange(0, 10, 1),
#     'y': np.array([2, 4, 6, 8, 10, 8, 6, 4, 2, 0])
# }
#
# # Create a DataFrame from the data
# df = pd.DataFrame(data)
#
# # Use Streamlit to create a line chart
# st.line_chart(data=df, x='x', y='y', width=0, height=0, use_container_width=True)
#
# # Add a title and axis labels
# st.title("Example Line Chart")
# # st.xlabel("X-axis")
# # st.ylabel("Y-axis")



# st.title("Progress Bar Example")
#
# # Button to start the process
# if st.button("Start Process"):
#     with st.spinner("Running the process..."):
#         progress_bar = st.progress(0)  # Initialize the progress bar
#         for i in range(51):
#             time.sleep(0.1)  # Simulate some processing time
#             progress_bar.progress(i, text=f"{i}")  # Update the progress bar
#         progress_bar.empty()
#
#         st.success("Process completed!")  # Display a success message once the process is finished



# # Function that simulates a process
# def simulate_process(value):
#     st.write(f"Simulating process with value: {value}")
#     # Replace this with your actual process code
#     # ...
#
# # Streamlit app
# st.title("Process Trigger with Slider")
#
# # Add a slider widget
# selected_value = st.slider("Select a value", min_value=0, max_value=100, value=50)
#
# # Check if the slider value has changed
# if "prev_value" not in st.session_state:
#     st.session_state.prev_value = selected_value
#
# if st.session_state.prev_value != selected_value:
#     st.session_state.prev_value = selected_value
#     simulate_process(selected_value)
#
