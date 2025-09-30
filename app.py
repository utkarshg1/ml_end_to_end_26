# Import necessary libraries
import streamlit as st
from inference import load_model, get_predicitons

# Initialize a blank streamlit page
st.set_page_config(page_title="End to End ML Project")

# Load the model object
model = load_model()

# Add the title
st.title("End to End ML Project")
st.subheader("by Utkarsh Gaikwad")

# Add user inputs
sep_len = st.number_input("Sepal Length : ", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width : ", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length : ", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width : ", min_value=0.00, step=0.01)

# Create a button to predict the results
button = st.button("Predict", type="primary")

# If button is clicked then predict results
if button:
    preds, probs = get_predicitons(model, sep_len, sep_wid, pet_len, pet_wid)
    st.subheader("Predictions : ")
    st.subheader(f"Species Predicted : {preds}")
    st.dataframe(data=probs)
    st.bar_chart(probs.T)
