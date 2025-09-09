import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
@st.cache_data
def load_data(path="telco_churn.csv"):
    df = pd.read_csv(path)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

st.title("Customer Churn Dashboard")

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")
    tenure_range = st.slider("Tenure (months)", int(df.tenure.min()), int(df.tenure.max()), (int(df.tenure.min()), int(df.tenure.max())))
    gender_options = st.multiselect("Gender", options=df["gender"].unique(), default=list(df["gender"].unique()))
    contract_options = st.multiselect("Contract Type", options=df["Contract"].unique(), default=list(df["Contract"].unique()))
    monthly_range = st.slider("Monthly Charges", float(df.MonthlyCharges.min()), float(df.MonthlyCharges.max()),
                              (float(df.MonthlyCharges.min()), float(df.MonthlyCharges.max())))

# Apply filters
df_filtered = df[
    (df["tenure"].between(*tenure_range)) &
    (df["gender"].isin(gender_options)) &
    (df["Contract"].isin(contract_options)) &
    (df["MonthlyCharges"].between(*monthly_range))
]

# --- First row: KPIs ---
kpi_col1, kpi_col2 = st.columns(2)
with kpi_col1:
    st.metric("Churn Rate", f"{df_filtered['Churn'].mean():.1%}")
with kpi_col2:
    st.metric("Avg Tenure", f"{df_filtered['tenure'].mean():.1f} months")

# --- Second row: core charts ---
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig1 = px.histogram(df_filtered, x="MonthlyCharges", color="Churn", barmode="overlay",
                        title="Monthly Charges by Churn Status")
    st.plotly_chart(fig1)

with chart_col2:
    fig2 = px.bar(df_filtered.groupby("Contract")["Churn"].mean().reset_index(),
                  x="Contract", y="Churn", title="Churn Rate by Contract Type")
    st.plotly_chart(fig2)

# --- Third row: new charts ---
chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    if "PaymentMethod" in df.columns:
        fig3 = px.bar(df_filtered.groupby("PaymentMethod")["Churn"].mean().reset_index(),
                      x="PaymentMethod", y="Churn", title="Churn Rate by Payment Method")
        st.plotly_chart(fig3)

with chart_col4:
    if "InternetService" in df.columns:
        fig4 = px.bar(df_filtered.groupby("InternetService")["Churn"].mean().reset_index(),
                      x="InternetService", y="Churn", title="Churn Rate by Internet Service")
        st.plotly_chart(fig4)

# --- Optional scatter plot below ---
if st.checkbox("Show scatter: MonthlyCharges vs Tenure"):
    st.subheader("Monthly Charges vs Tenure")
    fig5 = px.scatter(df_filtered, x="tenure", y="MonthlyCharges", color="Churn")
    st.plotly_chart(fig5)

# --- Download filtered data ---
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=df_filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_churn_data.csv",
    mime="text/csv",
)

# --- Train ML model ---
if st.button("Train Random Forest"):
    X = pd.get_dummies(df_filtered.drop(columns=["customerID", "Churn"]), drop_first=True)
    y = df_filtered["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.subheader("Model Performance")
    st.text(classification_report(y_test, preds))

    # Feature importance
    st.subheader("Top 10 Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
    fig6, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    st.pyplot(fig6)