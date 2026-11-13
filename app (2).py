import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="ML Classification Dashboard", page_icon="📊", layout="wide")

st.title("📊 Machine Learning Classification Dashboard")
st.write(
    "Upload a CSV dataset, select the target column, compare five classification "
    "algorithms, evaluate them, and optionally apply PCA."
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin.")
    st.markdown("""
    **Suggested dataset:** IBM HR Analytics – Employee Attrition.

    For that dataset, choose `Attrition` as the target column. The app also works
    with other labelled classification datasets.
    """)
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("Your dataset must contain at least two columns and one data row.")
    st.stop()

st.subheader("1. Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)
with st.expander("Column data types and missing values"):
    info_df = pd.DataFrame({
        "Data type": df.dtypes.astype(str),
        "Missing values": df.isna().sum(),
        "Unique values": df.nunique(dropna=True)
    })
    st.dataframe(info_df, use_container_width=True)

st.subheader("2. Select prediction target")
target = st.selectbox("Which column do you want to predict?", options=list(df.columns))
drop_candidates = [c for c in df.columns if c != target and
                   (c.lower() == "id" or c.lower().endswith("_id"))]
drop_columns = st.multiselect(
    "Optional: remove identifier or irrelevant columns",
    options=[c for c in df.columns if c != target],
    default=drop_candidates
)

work = df.dropna(subset=[target]).copy()
X = work.drop(columns=[target] + drop_columns)
y_raw = work[target].astype(str)

# Remove columns with no usable information.
constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)

if X.shape[1] == 0:
    st.error("No predictor columns remain. Remove fewer columns and try again.")
    st.stop()

class_counts = y_raw.value_counts()
if y_raw.nunique() < 2:
    st.error("The target column must contain at least two classes.")
    st.stop()
if y_raw.nunique() > 30:
    st.warning(
        "Your target has more than 30 unique values. This app is designed for "
        "classification; check that you selected a categorical target."
    )

st.write("Target class distribution")
st.bar_chart(class_counts)

test_size = st.sidebar.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
use_pca = st.sidebar.checkbox("Also evaluate models with PCA", value=True)
run_cv = st.sidebar.checkbox("Run 5-fold cross-validation", value=True)

# Encode target labels consistently. The original class names are retained for display.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
class_names = list(label_encoder.classes_)

if len(y) < 10:
    st.error("At least 10 labelled rows are recommended for a meaningful train/test evaluation.")
    st.stop()

min_class_count = int(pd.Series(y).value_counts().min())
stratify_arg = y if min_class_count >= 2 else None
if stratify_arg is None:
    st.warning("At least one target class has only one row, so the split cannot be stratified.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42, stratify=stratify_arg
)

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

transformers = []
if numeric_features:
    transformers.append(("num", numeric_transformer, numeric_features))
if categorical_features:
    transformers.append(("cat", categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Support Vector Classifier": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

st.subheader("3. Train and compare classification models")
if st.button("🚀 Train models", type="primary", use_container_width=True):
    results = []
    fitted_models = {}
    predictions = {}

    progress = st.progress(0)
    status = st.empty()

    for idx, (name, estimator) in enumerate(models.items(), start=1):
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            fitted_models[name] = pipe
            predictions[name] = pred
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                "F1-score (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0)
            })
        except Exception as exc:
            st.warning(f"{name} could not be trained: {exc}")
        progress.progress(idx / len(models))
        status.write(f"Processed {name}")

    if not results:
        st.error("None of the models could be trained. Check your dataset and target column.")
        st.stop()

    results_df = pd.DataFrame(results).set_index("Model").sort_values("F1-score (weighted)", ascending=False)
    st.session_state["results_df"] = results_df
    st.session_state["fitted_models"] = fitted_models
    st.session_state["predictions"] = predictions
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["class_names"] = class_names
    st.session_state["target"] = target
    st.session_state["X_columns"] = list(X.columns)

    st.success("Model training completed.")
    st.subheader("Test-set performance")
    st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(results_df[["Accuracy", "F1-score (weighted)"]])

    best_model_name = results_df.index[0]
    st.metric("Best model by weighted F1-score", best_model_name)

    selected_model = st.selectbox("Choose a model for detailed evaluation", list(fitted_models.keys()))
    selected_pred = predictions[selected_model]
    cm = confusion_matrix(y_test, selected_pred)
    st.write(f"Confusion matrix – {selected_model}")
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    st.write("Classification report")
    report = classification_report(
        y_test, selected_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    if run_cv:
        st.subheader("4. Five-fold cross-validation")
        if min_class_count >= 5:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_rows = []
            for name, estimator in models.items():
                cv_pipe = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator)
                ])
                try:
                    scores = cross_validate(
                        cv_pipe, X, y, cv=cv,
                        scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
                    )
                    cv_rows.append({
                        "Model": name,
                        "CV Accuracy (mean)": scores["test_accuracy"].mean(),
                        "CV Precision (mean)": scores["test_precision_weighted"].mean(),
                        "CV Recall (mean)": scores["test_recall_weighted"].mean(),
                        "CV F1 (mean)": scores["test_f1_weighted"].mean()
                    })
                except Exception as exc:
                    st.warning(f"Cross-validation skipped for {name}: {exc}")
            if cv_rows:
                cv_df = pd.DataFrame(cv_rows).set_index("Model").sort_values("CV F1 (mean)", ascending=False)
                st.dataframe(cv_df.style.format("{:.3f}"), use_container_width=True)
        else:
            st.info("Five-fold stratified cross-validation needs at least five rows in every target class.")

    if use_pca:
        st.subheader("5. PCA comparison")
        st.caption(
            "PCA is fitted inside each training pipeline to avoid fitting the dimensionality "
            "reduction step on the test data."
        )
        pca_rows = []
        pca_models = {}
        for name, estimator in models.items():
            pca_pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("pca", PCA(n_components=0.95)),
                ("model", estimator)
            ])
            try:
                pca_pipe.fit(X_train, y_train)
                pca_pred = pca_pipe.predict(X_test)
                pca_models[name] = pca_pipe
                pca_rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pca_pred),
                    "Precision (weighted)": precision_score(y_test, pca_pred, average="weighted", zero_division=0),
                    "Recall (weighted)": recall_score(y_test, pca_pred, average="weighted", zero_division=0),
                    "F1-score (weighted)": f1_score(y_test, pca_pred, average="weighted", zero_division=0),
                    "PCA components": pca_pipe.named_steps["pca"].n_components_
                })
            except Exception as exc:
                st.warning(f"PCA version of {name} could not be trained: {exc}")
        if pca_rows:
            pca_df = pd.DataFrame(pca_rows).set_index("Model").sort_values("F1-score (weighted)", ascending=False)
            st.dataframe(pca_df.style.format({
                "Accuracy": "{:.3f}",
                "Precision (weighted)": "{:.3f}",
                "Recall (weighted)": "{:.3f}",
                "F1-score (weighted)": "{:.3f}"
            }), use_container_width=True)
            st.session_state["pca_results_df"] = pca_df
            st.session_state["pca_models"] = pca_models

    st.subheader("6. Download results")
    st.download_button(
        "Download model comparison (CSV)",
        data=results_df.to_csv().encode("utf-8"),
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, scikit-learn, Pandas and Streamlit.")
