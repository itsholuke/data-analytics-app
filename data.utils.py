import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return clean_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df
