import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ==== Page Config ====
st.set_page_config(page_title="💼 Salary Predictor", layout="wide", page_icon="💰")

# ==== Load model & encoders ====
model = pickle.load(open("salary_model.pkl", "rb"))
le_G = pickle.load(open("le_gender.pkl", "rb"))
le_E = pickle.load(open("le_edu.pkl", "rb"))
le_J = pickle.load(open("le_job.pkl", "rb"))
le_L = pickle.load(open("le_loc.pkl", "rb"))

# ==== Sidebar ====
with st.sidebar:
    st.image("https://static.vecteezy.com/system/resources/previews/020/716/201/large_2x/illustration-of-business-man-doing-data-analysis-using-magnifying-glass-data-analytics-makes-predictions-of-future-business-free-png.png", width=160)
    st.title("Edunet Foundation-IBM SkillsBuild")
    st.markdown("""
    **Intern:** Ashika Sunder  
    **Project:** Employee Salary Prediction  
    **Tech:** Streamlit + ML + Seaborn + Plotly
    ---
    💡 *Enter employee details and get the predicted monthly salary instantly.*
    """)

    st.markdown("##### 🔗 [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)")

# ==== Title ====
st.markdown("""
    <h1 style='text-align: center; color: indigo;'>🎯 Employee Salary Prediction </h1>
    <hr style='border: 1px solid #ddd;'>
    """, unsafe_allow_html=True)

# ==== Form Input ====
st.markdown("### 🧾 Enter Employee Details Below")

with st.form(key="prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", le_G.classes_)
        education = st.selectbox("Education", le_E.classes_)
    with col2:
        experience = st.slider("Experience (Years)", 0, 40, 3)
        job = st.selectbox("Job Title", le_J.classes_)
        location = st.selectbox("Location", le_L.classes_)

    submit = st.form_submit_button("🚀 Predict Salary")

# ==== Predict Button Logic ====
if submit:
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": le_G.transform([gender])[0],
        "Education": le_E.transform([education])[0],
        "Experience": experience,
        "Job_Title": le_J.transform([job])[0],
        "Location": le_L.transform([location])[0]
    }])

    salary = model.predict(input_data)[0]
    st.success(f"💰 Estimated Monthly Salary: ₹ {salary:,.2f}")

    # Download option
    input_data["Predicted_Salary"] = salary
    csv = input_data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Result as CSV", data=csv, file_name="salary_prediction.csv", mime="text/csv")

# ==== Load Data for Insights ====
df = pd.read_csv("employee_cleaned.csv")

# ==== Tabs for Insights ====
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Seaborn Charts", "📈 Interactive Charts", "🧠 Correlation Heatmap", "📤 Batch Prediction"])

# ==== Tab 1: Static Seaborn Charts ====
with tab1:
    st.subheader("Experience vs Salary (by Education)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x="Experience", y="Salary", hue="Education", palette="cool", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Average Salary by Job Title")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    job_salary = df.groupby("Job_Title")["Salary"].mean().sort_values(ascending=False).head(10)
    job_salary.plot(kind="bar", ax=ax2, color="purple")
    ax2.set_ylabel("Avg Salary")
    st.pyplot(fig2)

# ==== Tab 2: Plotly Interactive Charts ====
with tab2:
    st.subheader("Interactive: Salary vs Age")
    fig3 = px.scatter(df, x="Age", y="Salary", color="Education", size="Experience",
                      title="Salary by Age & Education", hover_data=["Job_Title"])
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Interactive: Experience Distribution")
    fig4 = px.histogram(df, x="Experience", nbins=30, title="Experience Distribution")
    st.plotly_chart(fig4, use_container_width=True)

# ==== Tab 3: Correlation Heatmap ====
with tab3:
    st.subheader("📊 Correlation Heatmap")
    corr_df = df[["Age", "Experience", "Salary"]].copy()
    corr = corr_df.corr()
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="magma", ax=ax5)
    st.pyplot(fig5)

# ==== Tab 4: Batch Prediction ====
with tab4:
    st.subheader("📤 Upload CSV for Bulk Salary Prediction")
    st.markdown("**Columns Required:** Age, Gender, Education, Experience, Job_Title, Location")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)

        try:
            df_input["Gender"] = le_G.transform(df_input["Gender"])
            df_input["Education"] = le_E.transform(df_input["Education"])
            df_input["Job_Title"] = le_J.transform(df_input["Job_Title"])
            df_input["Location"] = le_L.transform(df_input["Location"])

            preds = model.predict(df_input)
            df_input["Predicted_Salary"] = preds

            st.write("✅ Batch Predictions:")
            st.dataframe(df_input)

            csv_out = df_input.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Predicted CSV", data=csv_out, file_name="batch_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ==== Custom Styling ====
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==== Footer ====
st.markdown("""
    <br><hr>
    <p style='text-align: center;'> Ashika Sunder | ashikasunder.gmail.com</p>
    """, unsafe_allow_html=True)
