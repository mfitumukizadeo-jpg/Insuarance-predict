"""
INSURANCE CHARGES PREDICTION
Professional Machine Learning Dashboard

"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# MODERN EXECUTIVE DASHBOARD THEME
# ============================================================

st.markdown("""
<style>

/* Main App background and font styling */
.stApp {
    background-color: #0F172A;
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1300px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Typography headers with gradient/vibrant accents */
h1 {
    color: #818CF8 !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em;
}

h2 {
    color: #A5B4FC !important;
    font-weight: 700 !important;
}

h3 {
    color: #38BDF8 !important;
    font-weight: 600 !important;
}

p, label, li {
    color: #94A3B8;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.875rem !important;
}

[data-testid="stMetricValue"] {
    color: #818CF8 !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    color: #34D399 !important;
}

/* Dataframe & Expanders */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 14px;
    background-color: #1E293B;
}

[data-testid="stExpander"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 14px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

.stAlert {
    border-radius: 12px;
}

hr {
    border-color: #334155;
    margin-top: 2rem;
    margin-bottom: 2rem;
}

.stCaption {
    color: #64748B;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Insurance Analytics Dashboard")

st.markdown("""
Analyze medical insurance data, explore important features,
evaluate a Linear Regression model, and estimate insurance charges.
""")

st.markdown("""
<div style="
    background: linear-gradient(90deg, #1E293B 0%, #111827 100%);
    padding: 18px 22px;
    border-radius: 14px;
    border: 1px solid #334155;
    border-left: 6px solid #6366F1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
">
    <span style="color:#818CF8;font-weight:700;font-size:1.05rem;">
        MACHINE LEARNING PRACTICAL
    </span>
    <br>
    <span style="color:#94A3B8;font-size:0.95rem;">
        Algorithm: Linear Regression | Dataset: Medical Insurance
        | Questions 6–13
    </span>
</div>
""", unsafe_allow_html=True)

st.write("")


# ============================================================
# DATASET
# ============================================================

DATA_PATH = Path("insurance.csv")

if not DATA_PATH.exists():
    st.error(
        "insurance.csv was not found. "
        "Place it beside this Python file in your GitHub repository."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = [
    "age", "sex", "bmi", "children",
    "smoker", "region", "charges"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

df = df[required_columns].dropna().copy()


# ============================================================
# LOAD AND DISPLAY DATASET
# ============================================================

st.divider()

st.header("📊  Dataset Exploration")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Total Records", f"{len(df):,}")
metric2.metric("Total Features", "6")
metric3.metric("Target Variable", "Charges")

with st.expander("View First Five Rows", expanded=True):
    st.dataframe(
        df.head(),
        use_container_width=True,
        hide_index=True
    )

with st.expander("View Summary Statistics"):
    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ============================================================
# ONE-HOT ENCODING
# ============================================================

st.divider()

st.header("🔠  Categorical Variables")

st.markdown("""
Categorical features are converted into numerical features
using One-Hot Encoding.
""")

df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

with st.expander("View Encoded Dataset", expanded=True):
    st.dataframe(
        df_encoded.head(),
        use_container_width=True,
        hide_index=True
    )

    st.caption("Encoded feature names:")
    st.write(df_encoded.columns.tolist())


# ============================================================
# TRAINING AND TESTING SPLIT
# ============================================================

st.divider()

st.header("📂 Training and Testing Split")

X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Dataset · 80%",
        f"{len(X_train):,} records"
    )

with col2:
    st.metric(
        "Testing Dataset · 20%",
        f"{len(X_test):,} records"
    )


# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

st.divider()

st.header("🤖  Model Training")

model = LinearRegression()
model.fit(X_train, y_train)

st.success("Linear Regression model trained successfully.")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

with st.expander("View Model Coefficients", expanded=True):
    st.dataframe(
        coefficients,
        use_container_width=True,
        hide_index=True
    )

st.markdown("### Model Intercept")

st.code(f"{model.intercept_:.4f}")


# ============================================================
# REGRESSION EQUATION
# ============================================================

st.divider()

st.header("🧮  Regression Equation")

equation = f"charges = {model.intercept_:.4f}"

for feature, coefficient in zip(X.columns, model.coef_):
    sign = "+" if coefficient >= 0 else "-"
    equation += (
        f" {sign} {abs(coefficient):.4f}({feature})"
    )

with st.expander("View Complete Equation", expanded=True):
    st.code(equation, language="text")


# ============================================================
# MODEL EVALUATION
# ============================================================

st.divider()

st.header("📈  Model Evaluation")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"${mae:,.2f}"
    )

with col2:
    st.metric(
        "MSE",
        f"${mse:,.2f}"
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2 * 100:.2f}%"
    )

st.caption(
    "R² represents the proportion of variation explained by the model. "
    "It is not classification accuracy."
)


# ============================================================
# ACTUAL VS PREDICTED REGRESSION PLOT
# ============================================================

st.divider()

st.header("📉  Regression Plot")

fig1, ax1 = plt.subplots(figsize=(10, 6))

fig1.patch.set_facecolor("#0F172A")
ax1.set_facecolor("#1E293B")

ax1.scatter(
    y_test,
    y_pred,
    color="#38BDF8",
    alpha=0.75,
    s=45,
    label="Predicted charges"
)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

ax1.plot(
    [line_min, line_max],
    [line_min, line_max],
    color="#F43F5E",
    linestyle="--",
    linewidth=2.5,
    label="Ideal prediction"
)

ax1.set_xlabel(
    "Actual Insurance Charges ($)",
    color="#E2E8F0"
)

ax1.set_ylabel(
    "Predicted Insurance Charges ($)",
    color="#E2E8F0"
)

ax1.set_title(
    "Actual vs Predicted Insurance Charges",
    color="#818CF8",
    fontsize=14,
    fontweight="bold"
)

ax1.tick_params(colors="#94A3B8")

for spine in ax1.spines.values():
    spine.set_color("#334155")

ax1.grid(color="#334155", alpha=0.4)

ax1.legend(
    facecolor="#1E293B",
    edgecolor="#334155",
    labelcolor="#E2E8F0"
)

fig1.tight_layout()

st.pyplot(fig1)

plt.close(fig1)


# ============================================================
# CONFUSION MATRIX EXPLANATION
# ============================================================

st.markdown("### ❓ Confusion Matrix Explanation")

st.info("""
A confusion matrix is suitable for classification models.
Linear Regression predicts continuous insurance charges.
Therefore, a confusion matrix is not directly appropriate
for this regression task.
""")


# ============================================================
# PREDICT CHARGES
# ============================================================

st.divider()

st.header("💰 Insurance Charges Prediction")

st.markdown("""
### Individual Information
""")

new_person = pd.DataFrame([{
    "age": 19,
    "sex": "female",
    "bmi": 27.9,
    "children": 0,
    "smoker": "yes",
    "region": "southwest"
}])

st.dataframe(
    new_person,
    use_container_width=True,
    hide_index=True
)

new_person_encoded = pd.get_dummies(
    new_person,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

new_person_encoded = new_person_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

predicted_charge = model.predict(
    new_person_encoded
)[0]

st.markdown("""
<div style="
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #34D399;
    border-left: 6px solid #34D399;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
">
    <div style="
        color: #34D399;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.05em;
    ">
        ESTIMATED INSURANCE CHARGES
    </div>
    <div style="
        color: #F8FAFC;
        font-size: 40px;
        font-weight: 800;
        margin-top: 8px;
        letter-spacing: -0.025em;
    ">
        $""" + f"{predicted_charge:,.2f}" + """
    </div>
    <div style="color: #64748B; font-size: 13px; margin-top: 4px;">
        Linear Regression Prediction Model
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#64748B;
    font-size:13px;
    padding:10px;
">
    INSURANCE ANALYTICS DASHBOARD
    <br>
    Machine Learning Practical · Linear Regression
</div>
""", unsafe_allow_html=True)
