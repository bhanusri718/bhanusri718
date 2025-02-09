import streamlit as st
import numpy as np
import pickle

with open(r'C:\Users\YESHWANTH\machine learning\kmeans_model.pkl','rb') as file:
    model = pickle.load(file)

print("Model loaded successfully!")

st.title("purpose of the customers")
st.write("this app predicts the customers spending category of a customer using kmeans model. ") 

Annual_income = st.number_input("Annual income(k$):", min_value=18.0, max_value=200.0, value=18.0, step=0.5)
spending=st.number_input("Spending score(1-100)",min_value=1,max_value=100, step=100)
# When the button is clicked, make predictions
if st.button("predict customers Category"):
     # Make a prediction using the trained model
     input_data=np.array([[Annual_income,spending]])
     prediction = model.predict(input_data)
     st.success(f"the predicted  category for an  annual income of {Annual_income}K and a spending score of {spending} is: {prediction[0]}")    
  
     # Display information about the model
st.write("the model was trained using the dataset of Mall_Customers")
