import streamlit as st
from data_utils import load_data
from eda import run_eda
from predictive_modeling import run_modeling
from prescriptive import run_prescriptive

st.set_page_config(page_title="Business School Analytics App", layout="wide")
st.title("📊 Business School Data Analytics Platform")

st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Dataset successfully loaded!")

    st.sidebar.header("2. Choose Analysis")
    analysis_type = st.sidebar.radio("Select analysis type:",
                                     ("Descriptive Analytics", "Predictive Modeling", "Prescriptive Analytics"))

    if analysis_type == "Descriptive Analytics":
        run_eda(df)
    elif analysis_type == "Predictive Modeling":
        run_modeling(df)
    elif analysis_type == "Prescriptive Analytics":
        run_prescriptive(df)
else:
    st.info("Please upload a CSV file to begin.")
