import streamlit as st
import pickle
import numpy as np

# Local image path
background_image = r"C:\Users\HP\.vscode\.vscode\.vscode\Machine Learning\20240916135849_home.avif"

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

# Title of the Streamlit app
st.title("House Price Prediction App")

# Brief description
st.write("This app predicts the price of a house based on its size using a simple linear regression model.")

# Load the saved model
try:
    with open(r"C:\Users\HP\.vscode\.vscode\.vscode\LR_House_pred.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()

# Input widget for user to enter house size
house_size = st.number_input("Enter the size of the house (in sqft):", min_value=0.0, max_value=9999999.0, step=0.05)

# Predict button and logic
if st.button("Predict Price"):
    if house_size > 0:
        size_input = np.array([[house_size]])
        try:
            prediction = model.predict(size_input)
            st.success(f"The predicted price for a house of size {house_size:.2f} sqft is ₹{prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter a valid house size greater than 0.")

# Additional footer
st.write("The model was trained using a dataset of house prices and sizes.")
st.write("Built with ❤️ by Sumit Chouhan.")
