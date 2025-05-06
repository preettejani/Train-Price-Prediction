import streamlit as st
import pandas as pd
import pickle as pk

st.title("Flight Price Prediction App")

model = pk.load(open('model.pkl', 'rb'))

st.write(" Model Loaded Successfully!")

airline = st.sidebar.selectbox("Select Airline", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet','Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',  'Vistara Premium economy', 'Multiple carriers Premium economy', 'Trujet'])
source = st.sidebar.selectbox("Source", ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
destination = st.sidebar.selectbox("Destination", ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
total_stops = st.sidebar.selectbox("Total Stops", [0, 1, 2, 3, 4])
journey_day = st.sidebar.slider("Journey Day", 1, 31)
journey_month = st.sidebar.slider("Journey Month", 1, 12)
dep_hour = st.sidebar.slider("Departure Hour", 0, 23)
dep_minute = st.sidebar.slider("Departure Minute", 0, 59)
arrival_hour = st.sidebar.slider("Arrival Hour", 0, 23)
arrival_minute = st.sidebar.slider("Arrival Minute", 0, 59)
duration_hours = st.sidebar.slider("Duration Hours", 0, 20)
duration_minutes = st.sidebar.slider("Duration Minutes", 0, 59)

user_data = pd.DataFrame({
    'Airline': [airline],
    'Source': [source],
    'Destination': [destination],
    'Total_Stops': [total_stops],
    'Journey_Day': [journey_day],
    'Journey_Month': [journey_month],
    'Dep_Hour': [dep_hour],
    'Dep_Minute': [dep_minute],
    'Arrival_Hour': [arrival_hour],
    'Arrival_Minute': [arrival_minute],
    'Duration_Hours': [duration_hours],
    'Duration_Minutes': [duration_minutes]
})

st.write("User Input Data:", user_data)
user_data_transformed = model['preprocessor'].transform(user_data)
predicted_price = model['regressor'].predict(user_data_transformed)
st.success(f"Estimated Flight Price: ₹{predicted_price[0]:,.2f}")