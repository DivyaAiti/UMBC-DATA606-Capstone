import streamlit as st
import joblib
import pandas as pd

# Set the main title for the Streamlit app
st.title("Predicting the Severity of Road Traffic Accidents")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_gradient_boosting_model.pkl')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()  # Stop execution if the model cannot be loaded

model = load_model()

# Define input features based on the dataset
required_features = [
    'casualty_class', 'sex_of_casualty', 'age_of_casualty', 'age_band_of_casualty',
    'pedestrian_location', 'pedestrian_movement', 'car_passenger', 'bus_or_coach_passenger',
    'pedestrian_road_maintenance_worker', 'casualty_type', 'casualty_home_area_type',
    'casualty_imd_decile', 'lsoa_of_casualty'
]

# Function to get user input for model prediction
def get_user_input():
    # Collect numeric inputs for each feature, aligning with the dataset's structure
    age_of_casualty = st.sidebar.slider("Age of Casualty", 0, 100, 30)
    sex_of_casualty = st.sidebar.selectbox("Sex of Casualty (1=Male, 2=Female)", [1, 2])
    pedestrian_location = st.sidebar.selectbox("Pedestrian Location (0=Other, 1=Crossing)", [0, 1])
    casualty_class = st.sidebar.selectbox("Casualty Class (0=Pedestrian, 1=Driver, 2=Passenger)", [0, 1, 2])
    
    # Additional fields based on dataset encoding
    age_band_of_casualty = st.sidebar.slider("Age Band of Casualty (0-10 scale)", 0, 10, 5)
    pedestrian_movement = st.sidebar.selectbox("Pedestrian Movement (0=None, 1=Walking, 2=Running)", [0, 1, 2])
    car_passenger = st.sidebar.selectbox("Car Passenger (0=No, 1=Yes)", [0, 1])
    bus_or_coach_passenger = st.sidebar.selectbox("Bus or Coach Passenger (0=No, 1=Yes)", [0, 1])
    pedestrian_road_maintenance_worker = st.sidebar.selectbox("Pedestrian Road Maintenance Worker (0=No, 1=Yes)", [0, 1])
    casualty_type = st.sidebar.selectbox("Casualty Type (0=Pedestrian, 1=Cyclist, 2=Motorcyclist, 3=Driver, 4=Passenger)", [0, 1, 2, 3, 4])
    casualty_home_area_type = st.sidebar.selectbox("Casualty Home Area Type (0=Rural, 1=Urban)", [0, 1])
    casualty_imd_decile = st.sidebar.slider("Casualty IMD Decile (1-10 scale)", 1, 10, 5)
    lsoa_of_casualty = st.sidebar.text_input("LSOA of Casualty", "E01033378")  # Replace as needed

    # Map inputs to match the dataset structure and convert to a DataFrame
    user_data = {
        "casualty_class": casualty_class,
        "sex_of_casualty": sex_of_casualty,
        "age_of_casualty": age_of_casualty,
        "age_band_of_casualty": age_band_of_casualty,
        "pedestrian_location": pedestrian_location,
        "pedestrian_movement": pedestrian_movement,
        "car_passenger": car_passenger,
        "bus_or_coach_passenger": bus_or_coach_passenger,
        "pedestrian_road_maintenance_worker": pedestrian_road_maintenance_worker,
        "casualty_type": casualty_type,
        "casualty_home_area_type": casualty_home_area_type,
        "casualty_imd_decile": casualty_imd_decile,
        "lsoa_of_casualty": 1 if lsoa_of_casualty == "E01033378" else 0  # Adjust based on actual LSOA encoding
    }

    # Convert user input into a DataFrame with required features
    user_input_df = pd.DataFrame([user_data], columns=required_features)
    return user_input_df

# Gather user input
input_data = get_user_input()

# Display user input
st.write("User Input Parameters:")
st.write(input_data)

# Predict severity
if st.button("Predict Severity"):
    try:
        prediction = model.predict(input_data)
        severity_mapping = {1: "Fatal", 2: "Serious", 3: "Slight"}
        severity = severity_mapping.get(prediction[0], "Unknown")
        st.write(f"Predicted Severity: **{severity}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
