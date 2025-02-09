import streamlit as st
import pickle
import numpy as np

model = pickle.load(open(r'C:\Users\YESHWANTH\machine learning\classifier.pkl',"rb"))

# Set the title of the Streamlit app
st.title("vehicle salesprediction App")
# Add a brief description
st.write("This app predicts the vehicle sales based on years of experience using a Knn classifier model.")

# Add input widget for user to enter age
Age =st.number_input("Age:", min_value=18.0, max_value=60.0, value=25.0, step=1.0)
salary =st.number_input("Estimated Salary",min_value=0, step=100)
# When the button is clicked, make predictions
if st.button("predict to buy a vahicle"):
     # Make a prediction using the trained model
     input_data=np.array([[Age, salary]])
     prediction = model.predict(input_data)
     
     st.success(f"the predicted vehicle for {Age} and {salary} is: {prediction[0]}")
 
     # Display information about the model
st.write("the model was trained using the dataset of social_Network_ads")
