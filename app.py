import streamlit as st
import pickle
import numpy as np

# 1. LOAD THE TRAINED MODEL
try:
    with open('placement_model.pkl', 'rb') as f:
        data = pickle.load(f)
    model = data["model"]
    le_gender = data["le_gender"]
    le_stream = data["le_stream"]
except FileNotFoundError:
    st.error("Model file not found! Please run 'train_model.py' first.")
    st.stop()

# 2. APP TITLE & DESCRIPTION
st.set_page_config(page_title="Placement Predictor", page_icon="🎓")
st.title("🎓 College Placement Predictor")
st.write("Enter your academic details below to check your placement probability.")

# 3. CREATE INPUT FORM
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # We use the classes_ from our saved encoder to populate the dropdown automatically
        gender = st.selectbox("Gender", le_gender.classes_)
        stream = st.selectbox("Stream", le_stream.classes_)
        cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.1, value=8.0)

    with col2:
        internships = st.number_input("Number of Internships", min_value=0, max_value=10, step=1, value=1)
        backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=10, step=1, value=0)
    
    # Submit Button
    submitted = st.form_submit_button("Predict Result")

# 4. PREDICTION LOGIC
if submitted:
    # Convert user input to numbers using the loaded encoders
    gender_encoded = le_gender.transform([gender])[0]
    stream_encoded = le_stream.transform([stream])[0]
    
    # Create the input array (Must match the order in train_model.py)
    # [Gender, Stream, CGPA, Internships, Backlogs]
    # CORRECT ORDER (Must match X.columns from training)
    user_input = np.array([[gender_encoded, cgpa, internships, backlogs, stream_encoded]])
    
    # Predict
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[0][1] # Probability of being placed

    # Display Result
    st.divider()
    if prediction[0] == 1:
        st.success(f"🎉 **Congratulations!** You are likely to get **PLACED**.")
        st.info(f"Placement Probability: {probability*100:.2f}%")
        st.balloons()
    else:
        st.error(f"⚠️ **Result:** You might struggle to get placed.")
        st.warning("Tip: Try increasing your CGPA or doing more internships!")