import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_eda(df):
    st.subheader("📈 Descriptive Analytics")

    if st.checkbox("Show basic info"):
        st.write(df.info())
        st.write(df.describe(include='all'))

    if st.checkbox("Show missing values"):
        st.write(df.isnull().sum())

    if st.checkbox("Show correlation heatmap"):
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if st.checkbox("Visualize distributions"):
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        selected_col = st.selectbox("Choose a numeric column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

    if st.checkbox("Run PCA (Dimension Reduction)"):
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] > 1:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
            st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
            st.scatter_chart(pca_df)
        else:
            st.warning("Need at least 2 numeric columns for PCA.")
