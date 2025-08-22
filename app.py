import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load assets
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")
X_test, y_test, y_pred = joblib.load("test_predictions.pkl")
original_data = pd.read_csv("Salary Data.csv")
original_data.dropna(inplace=True)

# --- Streamlit UI ---

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💸", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict your salary with AI quickly and accurately.</p><hr>", unsafe_allow_html=True)


st.markdown("""
    <style>
     body {
        background-color: #ff7f50;
    }
    .main {
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .css-1aumxhk {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("man-user.png", width=200)
    st.title("Employee Salary Predictor")
    st.markdown(""" Key factors responsible for prediction  :
    - Age
    - Gender
    - Qualification
    - Job Role
    - Years of Experience
    """)
    st.markdown("---")
   

st.markdown("<h2 style='text-align: center;'>🧾 Enter Employee Details</h2>", unsafe_allow_html=True)


with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Age", 18, 65, 25)
        gender = st.radio("👤 Gender", ["Male", "Female"])
        education = st.selectbox("🎓 Qualification", original_data['Education Level'].unique())

    with col2:
        job = st.selectbox("💼 Job Role", original_data['Job Title'].unique())
        exp = st.number_input("⏳ Years of Experience", 0, 40, 0)

    submitted = st.form_submit_button("✨ Predict Salary")

    if submitted:
        # Construct input data
        input_dict = {col: 0 for col in model_columns}
        input_dict["Age"] = age
        input_dict["Years of Experience"] = exp

        # Encoding inputs
        gender_encoded = original_data['Gender'].unique().tolist().index(gender)
        qualification_encoded = original_data['Education Level'].unique().tolist().index(education)
        job_title_encoded = original_data['Job Title'].unique().tolist().index(job)

        input_dict["Gender"] = gender_encoded
        input_dict["Qualification"] = qualification_encoded
        input_dict["Job Title"] = job_title_encoded

        X_input = pd.DataFrame([input_dict], columns=model_columns)

        # Validations
        if exp >= age or exp > (age - 20) or age < (exp + 18):
            st.error("⚠️ Invalid combination of age and experience!")
            st.stop()
        if education == "Master's" and age < 23:
            st.error("⚠️ Age & Qualification doesn't match! Please adjust.")
            st.stop()
        if education == "PhD" and age < 26:
            st.error("⚠️ Age & Qualification doesn't match! Please adjust.")
            st.stop()

        # Prediction
        salary = model.predict(X_input)[0]
        st.success(f"🎉 Great news! Your predicted monthly salary is : **₹{salary:,.2f}**")
        st.info("💡 Keep in mind, This prediction is based on available data and does not guarantee actual compensation.")
        

        # Save the report in session state
        report_text = f'''
        📄 Salary Prediction Report
        ----------------------------
        👤 Gender: {gender}
        🎂 Age: {age}
        🎓 Qualification: {education}
        💼 Job Role: {job}
        ⏳ Experience: {exp} years

        💸 Predicted Monthly Salary: ₹{salary:,.2f}

        Disclaimer:
        This prediction is based on available data and does not guarantee actual compensation.
        '''
        st.session_state['report_text'] = report_text
        st.session_state['show_download'] = True


# Display download button after form is submitted
if st.session_state.get('show_download'):
    st.download_button(
        label="📥 Download Report",
        data=st.session_state['report_text'],
        file_name="salary_prediction_report.txt",
        mime="text/plain"
    )

# Container for model performance box
with st.expander("📊 Model performance"):
    st.markdown(f"**Selected Model:** {model.__class__.__name__}")
    st.markdown(f"**Mean Absolute Error (MAE):** {abs(y_test - y_pred).mean():,.2f}")
    st.markdown(f"**Mean Squared Error (MSE):** {(y_test - y_pred).pow(2).mean():,.2f}")
    st.markdown(f"**R-squared (R²):** {model.score(X_test, y_test):.4f}")

# Scatter plot
with st.expander("📈 Scatter Plot: Actual vs Predicted Salary"):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Actual vs Predicted Salary")
    st.pyplot(fig)