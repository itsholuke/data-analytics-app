import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

def run_prescriptive(df):
    st.subheader("🔍 Prescriptive Analytics")

    st.markdown("""
    This section provides optimization-based suggestions and recommendations based on your data.
    Example use cases include maximizing student satisfaction, minimizing costs, or allocating resources efficiently.
    """)

    st.markdown("---")
    st.markdown("### Simple Linear Optimization Example")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns for this optimization demo.")
        return

    objective_col = st.selectbox("Select column to maximize (objective):", numeric_cols)
    constraint_col = st.selectbox("Select constraint column (e.g., cost/resource):", [col for col in numeric_cols if col != objective_col])
    budget = st.number_input("Enter resource limit (constraint value):", min_value=1.0, value=100.0)

    c = -df[objective_col].values
    A = [df[constraint_col].values]
    b = [budget]
    bounds = [(0, None) for _ in range(len(df))]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if result.success:
        st.success("Optimization succeeded!")
        decision_vars = result.x.round(2)
        df_result = df.copy()
        df_result['Decision_Variables'] = decision_vars
        st.dataframe(df_result[df_result['Decision_Variables'] > 0])
        st.write("Total Objective Achieved:", -result.fun)
    else:
        st.error("Optimization failed. Try adjusting your inputs.")
