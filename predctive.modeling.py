import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

def run_modeling(df):
    st.subheader("🤖 Predictive Modeling")

    target = st.selectbox("Select the target column:", df.columns)
    features = st.multiselect("Select feature columns:", [col for col in df.columns if col != target])

    if not features or not target:
        st.warning("Please select both features and a target variable.")
        return

    X = df[features]
    y = df[target]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    task_type = 'classification' if len(set(y)) < 20 and y.dtype in ['int64', 'int32'] else 'regression'
    st.write(f"Detected task type: **{task_type.capitalize()}**")

    models = {
        'classification': {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC()
        },
        'regression': {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'SVR': SVR()
        }
    }

    for name, model in models[task_type].items():
        st.markdown(f"### Model: {name}")
        model.fit(X_train, y_train)
        if task_type == 'classification':
            preds = model.predict(X_test)
            st.text(classification_report(y_test, preds))
            scores = cross_val_score(model, X, y, cv=5)
            st.write("Cross-validation Accuracy:", scores.mean())
        else:
            preds = model.predict(X_test)
            st.write("MSE:", mean_squared_error(y_test, preds))
            st.write("R² Score:", r2_score(y_test, preds))
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            st.write("Cross-validation R²:", scores.mean())
