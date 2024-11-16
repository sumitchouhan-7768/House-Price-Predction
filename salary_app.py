import streamlit as st 
import pickle
import numpy as np 
import sklearn

# Local image path
background_image = r"C:/Users/HP/.vscode/.vscode/.vscode/Machine Learning/salary.jpeg"

# Add custom CSS to set a background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('file:///{background_image}');
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# load the ssaved data
with open(r"C:\Users\HP\.vscode\.vscode\.vscode\Machine Learning\linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)
print(model)

# set the title of streamlit  app
st.title("Salary Predction App")

#add a breif description
st.write("This app predicts the salary based on years of experience using simple linear regression model")

#add input widget for user to enter year of experience
year_experience = st.number_input("enter number of Experience:",min_value=0.0,max_value=50.0,value=1.0,step=0.5)

# when the button is clicked , make predction
prediction = None

if st.button("Predict Salary"):
    experience_input = np.array([[year_experience]])
    prediction = model.predict(experience_input)
if prediction is not None:
    st.success(f"The predicted salary for {year_experience} years of experience is: ${prediction[0]:,.2f}")
else:
    st.warning("Please provide input and click on 'Predict Salary' to get a prediction.")
st.write("The model was trained using a dataset of salaries and years of experience.buit model by Sumit Chouhan")
