import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(
    page_title="Logistic Regression Learning Platform",
    page_icon="📈",
    layout="wide",
)

# Custom dark theme styling for dashboard UI
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        color: #f5f7fa;
        background-color: #0f172a;
    }
    .css-1d391kg, .css-18ni7ap, .css-e8sgji {
        background-color: #0f172a !important;
    }
    .css-1d391kg * {
        color: #f8fafc !important;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 1.8rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background: #020617;
    }
    .stButton>button, .stSelectbox>div>div>div>button, .stTextInput>div>div>input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }
    .stSlider>div>div>div>div {
        background: #1e293b !important;
    }
    .stMarkdown>div {
        color: #f8fafc !important;
    }
    .stMetricValue, .stMetricLabel {
        color: #f8fafc !important;
    }
    .card {
        padding: 1.6rem;
        border-radius: 1rem;
        background: #111827;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
        margin-bottom: 1.5rem;
    }
    .sb-sidebar {
        background: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Logistic Regression Dashboard")
st.sidebar.markdown("A step-by-step learning suite for Logistic Regression with data exploration, preprocessing, training, prediction, and evaluation.")

page = st.sidebar.radio(
    "Pipeline Steps",
    [
        "Theory & Math",
        "Dataset Input",
        "Data Preprocessing",
        "Exploratory Data Analysis",
        "Model Training",
        "Prediction",
        "Evaluation",
    ],
)

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
    st.session_state.processed_df = None
    st.session_state.target_column = None
    st.session_state.model = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.class_mapping = None
    st.session_state.target_warning = None
    st.session_state.preprocessing_settings = {
        "missing_strategy": "None",
        "encoding": "Label Encoding",
        "scaling": True,
    }


@st.cache_data
def load_csv_from_url(url: str):
    return pd.read_csv(url)


def detect_continuous_target(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        unique_values = series.dropna().unique()
        return len(unique_values) > 15
    return False


def preprocess_data(df: pd.DataFrame, target_column: str, missing_strategy: str, scaling: bool):
    data = df.copy()
    y = data[target_column].copy()
    X = data.drop(columns=[target_column])

    # Missing value handling
    if missing_strategy == "Mean":
        X = X.fillna(X.mean(numeric_only=True))
        for col in X.select_dtypes(include=["object", "category"]):
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "")
    elif missing_strategy == "Mode":
        X = X.fillna(X.mode().iloc[0])

    # Encoding categoricals
    encoders = {}
    for col in X.select_dtypes(include=["object", "category"]):
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        encoders[col] = encoder

    # If target is categorical and non-numeric, encode too
    if X.shape[0] > 0 and y.dtype == object:
        y = LabelEncoder().fit_transform(y.astype(str))

    # Scaling numeric features
    if scaling:
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    processed = pd.concat([X, y.rename(target_column)], axis=1)
    return processed


def _can_stratify(y: pd.Series) -> bool:
    unique_classes = np.unique(y)
    if len(unique_classes) <= 1:
        return False
    counts = pd.Series(y).value_counts()
    return counts.min() >= 2


def configure_model(X: pd.DataFrame, y: pd.Series, split_ratio: int):
    test_size = (100 - split_ratio) / 100
    stratify_target = y if _can_stratify(y) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_target
    )
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap", color="#f8fafc")
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#f8fafc")
    return fig


def plot_distribution(df: pd.DataFrame, columns: list[str]):
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 4 * len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for col, ax in zip(columns, axes):
        sns.histplot(df[col].dropna(), kde=True, color="#38bdf8", ax=ax)
        ax.set_title(f"Distribution of {col}", color="#f8fafc")
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        ax.set_facecolor("#020617")
        ax.tick_params(colors="#f8fafc")
    fig.patch.set_facecolor("#020617")
    return fig


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix", color="#f8fafc")
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    ax.tick_params(colors="#f8fafc")
    return fig


def plot_roc_curve(y_true, y_score, classes):
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(classes) == 2:
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        y_score_pos = y_score[:, 1]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score_pos)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#38bdf8", linewidth=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b")
        ax.set_title("ROC Curve", color="#f8fafc")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    else:
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        auc_values = []
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            auc_values.append(roc_auc)
            ax.plot(fpr, tpr, linewidth=2, label=f"Class {cls} (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b")
        ax.set_title("Multiclass ROC Curves", color="#f8fafc")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize="small")
    ax.set_facecolor("#020617")
    fig.patch.set_facecolor("#020617")
    ax.tick_params(colors="#f8fafc")
    return fig


def display_header(title: str, subtitle: str):
    st.markdown(f"<div class=\"card\"><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


with st.container():
    display_header(
        "Logistic Regression Learning Platform",
        "A modern dashboard-style application for uploading data, visualizing patterns, and training Logistic Regression models step by step.",
    )

if page == "Theory & Math":
    st.markdown("<div class=\"card\"><h2>Logistic Regression Theory</h2><p>Understand the math behind classification, the sigmoid activation, and how coefficients shape decision boundaries.</p></div>", unsafe_allow_html=True)
    st.markdown(
        "### What is Logistic Regression?\n"
        "Logistic Regression is a classification algorithm that models the probability that an input belongs to a particular class. It is widely used for binary classification tasks and can be extended to multiclass problems."
    )
    st.markdown(
        "### Key equation\n"
        "Logistic Regression uses the logistic function (sigmoid) to convert linear combinations into probabilities:"
    )
    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")
    st.markdown(
        "where the linear term is:"
    )
    st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")
    st.markdown(
        "### Probability and decision boundary\n"
        "The predicted class is determined by whether the output probability is above a threshold (usually 0.5). The decision boundary is defined by the linear equation:"
    )
    st.latex(r"\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n = 0")
    st.markdown(
        "### Odds and log-odds\n"
        "Logistic Regression models the log-odds of the positive class as a linear function of the features:"
    )
    st.latex(r"\log\left(\frac{P(y=1)}{1-P(y=1)}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n")
    st.markdown(
        "### Loss function\n"
        "Training finds coefficients that minimize the logistic loss (cross-entropy) over the training data:"
    )
    st.latex(r"L(\beta)= -\sum_{i=1}^{m} \left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]")
    st.markdown(
        "### Why scaling and encoding matter\n"
        "- **Scaling** ensures features on different ranges contribute proportionally to the coefficients.\n"
        "- **Encoding** turns categorical variables into numeric form so the model can compute the linear combination.\n"
        "- **Regularization** (not shown here) helps prevent overfitting by penalizing large coefficients."
    )
    st.markdown("---")
    st.subheader("Theory Quick Tips")
    st.markdown(
        "- If the target is continuous, we bin it into categories for classification.\n"
        "- Logistic Regression outputs probabilities, not raw labels.\n"
        "- The sigmoid function is always between 0 and 1, making it ideal for probability modeling.\n"
        "- Model coefficients indicate whether a feature increases or decreases the log-odds of the positive class."
    )

elif page == "Dataset Input":
    st.markdown("<div class=\"card\"><h2>1. Dataset Input Module</h2><p>Upload a CSV, load from URL, preview data, and choose your target variable.</p></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        url = st.text_input("Or paste a direct CSV URL")

        if uploaded_file is not None:
            try:
                st.session_state.raw_df = pd.read_csv(uploaded_file)
                st.session_state.target_column = None
                st.session_state.processed_df = None
                st.success("Dataset loaded from file.")
            except Exception as exc:
                st.error(f"Unable to read uploaded file: {exc}")

        if url and st.button("Load from URL"):
            try:
                st.session_state.raw_df = load_csv_from_url(url)
                st.session_state.target_column = None
                st.session_state.processed_df = None
                st.success("Dataset loaded from URL.")
            except Exception as exc:
                st.error(f"Unable to load dataset from URL: {exc}")

    with col2:
        st.markdown("### Dataset Status")
        if st.session_state.raw_df is None:
            st.info("No dataset loaded yet.")
        else:
            st.success("Dataset is available for preview and selection.")
            st.write(f"**Rows:** {st.session_state.raw_df.shape[0]}")
            st.write(f"**Columns:** {st.session_state.raw_df.shape[1]}")
            st.write(f"**Columns detected:** {', '.join(st.session_state.raw_df.columns[:8].astype(str))}...")

    if st.session_state.raw_df is not None:
        st.markdown("---")
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.raw_df.head(), use_container_width=True)

        st.markdown("### Data Types")
        types = st.session_state.raw_df.dtypes.astype(str).rename("Type").to_frame()
        st.dataframe(types, use_container_width=True)

        st.markdown("---")
        st.subheader("Target Column Selection")
        target_column = st.selectbox("Select the target column", st.session_state.raw_df.columns.tolist(), index=0)
        st.session_state.target_column = target_column
        st.write("Selected target:", f"**{target_column}**")

elif page == "Data Preprocessing":
    st.markdown("<div class=\"card\"><h2>2. Data Preprocessing Module</h2><p>Handle missing values, encode categories, and scale numeric features.</p></div>", unsafe_allow_html=True)
    if st.session_state.raw_df is None:
        st.warning("Please load a dataset first in the Dataset Input step.")
    else:
        with st.form("preprocessing_form"):
            missing_strategy = st.selectbox("Missing value handling", ["None", "Mean", "Mode"], index=["None", "Mean", "Mode"].index(st.session_state.preprocessing_settings["missing_strategy"]))
            scaling = st.checkbox("Apply StandardScaler", value=st.session_state.preprocessing_settings["scaling"])
            submit = st.form_submit_button("Apply preprocessing")

        if submit:
            st.session_state.preprocessing_settings["missing_strategy"] = missing_strategy
            st.session_state.preprocessing_settings["scaling"] = scaling
            processed = preprocess_data(
                st.session_state.raw_df,
                st.session_state.target_column,
                missing_strategy,
                scaling,
            )
            st.session_state.processed_df = processed
            st.success("Preprocessing completed successfully.")

        if st.session_state.processed_df is not None:
            st.markdown("### Before vs After")
            before_col, after_col = st.columns(2)
            with before_col:
                st.write("Original dataset head")
                st.dataframe(st.session_state.raw_df.head(), use_container_width=True)
            with after_col:
                st.write("Processed dataset head")
                st.dataframe(st.session_state.processed_df.head(), use_container_width=True)

            st.markdown("### Processed Dataset Summary")
            st.write(f"Shape: {st.session_state.processed_df.shape}")
            st.dataframe(st.session_state.processed_df.dtypes.astype(str).rename("Type").to_frame(), use_container_width=True)
        else:
            st.info("Adjust preprocessing settings and apply to see processed dataset preview.")

elif page == "Exploratory Data Analysis":
    st.markdown("<div class=\"card\"><h2>3. Exploratory Data Analysis (EDA)</h2><p>Visualize correlations, class balance, and feature distributions.</p></div>", unsafe_allow_html=True)
    if st.session_state.raw_df is None:
        st.warning("Please load a dataset first in the Dataset Input step.")
    else:
        df = st.session_state.raw_df
        if st.session_state.target_column is None:
            st.warning("Select a target column in Dataset Input before running EDA.")
        else:
            target_series = df[st.session_state.target_column]
            if detect_continuous_target(target_series):
                st.warning("Target appears continuous. Model training will bin this variable into classification groups.")

            st.subheader("Class Distribution")
            class_counts = target_series.value_counts(dropna=False)
            st.bar_chart(class_counts)

            st.subheader("Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                heatmap_fig = plot_correlation_heatmap(numeric_df)
                st.pyplot(heatmap_fig)
            else:
                st.info("Not enough numeric features for a correlation heatmap.")

            st.subheader("Feature Distribution Visualization")
            numeric_cols = numeric_df.columns.tolist()
            if numeric_cols:
                selected = st.multiselect("Choose numeric features to visualize", numeric_cols, default=numeric_cols[:3])
                if selected:
                    dist_fig = plot_distribution(df, selected[:4])
                    st.pyplot(dist_fig)
                else:
                    st.info("Select at least one numeric feature.")
            else:
                st.info("No numeric features available for distribution plots.")

elif page == "Model Training":
    st.markdown("<div class=\"card\"><h2>4. Model Training Module</h2><p>Train Logistic Regression and inspect coefficients, train-test split, and model status.</p></div>", unsafe_allow_html=True)
    if st.session_state.raw_df is None or st.session_state.target_column is None:
        st.warning("Please load a dataset and select a target column first.")
    else:
        df = st.session_state.raw_df.copy()
        target_column = st.session_state.target_column
        raw_target = df[target_column]
        y = raw_target

        if detect_continuous_target(raw_target):
            st.warning("Target is continuous. Converting into classification labels using quantile binning.")
            bins = st.slider("Number of bins for target conversion", 2, 5, 3)
            y = pd.qcut(raw_target.rank(method="first"), q=bins, labels=False, duplicates="drop")
            st.session_state.class_mapping = {i: f"Class {i}" for i in range(y.nunique())}
            st.write("Converted continuous target into", y.nunique(), "class labels.")
        else:
            if raw_target.dtype == object or pd.api.types.is_categorical_dtype(raw_target):
                y = LabelEncoder().fit_transform(raw_target.astype(str))
                st.session_state.class_mapping = {i: label for i, label in enumerate(np.unique(raw_target.astype(str)))}
            else:
                y = raw_target.astype(int)
                st.session_state.class_mapping = {i: str(i) for i in np.unique(y)}

        if st.button("Prepare and Train Logistic Regression"):
            if st.session_state.processed_df is None:
                processed = preprocess_data(
                    df,
                    target_column,
                    st.session_state.preprocessing_settings["missing_strategy"],
                    st.session_state.preprocessing_settings["scaling"],
                )
                st.session_state.processed_df = processed
            processed = st.session_state.processed_df
            X = processed.drop(columns=[target_column])
            y = processed[target_column].astype(int)
            split_ratio = st.slider("Training split percentage", 50, 90, 80)
            model, X_train, X_test, y_train, y_test = configure_model(X, y, split_ratio)
            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success("Model training completed.")

        if st.session_state.model is not None:
            st.subheader("Model Summary")
            st.write(f"**Algorithm:** Logistic Regression")
            st.write(f"**Training split:** {st.session_state.X_train.shape[0]} samples train, {st.session_state.X_test.shape[0]} samples test")
            coef_col, int_col = st.columns(2)
            with coef_col:
                st.write("### Coefficients")
                coeffs = pd.Series(st.session_state.model.coef_[0], index=st.session_state.X_train.columns)
                st.dataframe(coeffs.rename("Coefficient"), use_container_width=True)
            with int_col:
                st.write("### Intercept")
                st.write(st.session_state.model.intercept_[0])

            st.markdown("---")
            st.write("Model is ready for prediction and evaluation in the next steps.")
        else:
            st.info("Train the model to unlock prediction and evaluation modules.")

elif page == "Prediction":
    st.markdown("<div class=\"card\"><h2>5. Prediction Module</h2><p>Provide feature values and get predicted class probabilities.</p></div>", unsafe_allow_html=True)
    if st.session_state.model is None or st.session_state.processed_df is None:
        st.warning("Train the model first in the Model Training step.")
    else:
        X = st.session_state.processed_df.drop(columns=[st.session_state.target_column])
        st.subheader("Enter Feature Values")
        input_values = {}
        feature_cols = X.columns.tolist()
        input_cols = st.columns(3)
        for idx, feature in enumerate(feature_cols[:9]):
            with input_cols[idx % 3]:
                if pd.api.types.is_numeric_dtype(X[feature]):
                    input_values[feature] = st.number_input(feature, value=float(X[feature].median()))
                else:
                    input_values[feature] = st.text_input(feature, value="0")

        if st.button("Run Prediction"):
            sample = pd.DataFrame([input_values])
            sample = sample[X.columns]
            prediction = st.session_state.model.predict(sample)
            probabilities = st.session_state.model.predict_proba(sample)[0]
            predicted_label = prediction[0]
            label_name = st.session_state.class_mapping.get(predicted_label, str(predicted_label))
            st.success("Prediction complete.")
            st.write(f"**Predicted class:** {label_name}")
            proba_series = pd.Series(probabilities, index=[st.session_state.class_mapping.get(i, str(i)) for i in range(len(probabilities))])
            st.write("**Class probabilities:**")
            st.dataframe(proba_series.rename("Probability"), use_container_width=True)

elif page == "Evaluation":
    st.markdown("<div class=\"card\"><h2>6. Evaluation Module</h2><p>Inspect accuracy, precision, recall, F1-score, and the confusion matrix.</p></div>", unsafe_allow_html=True)
    if st.session_state.model is None or st.session_state.X_test is None or st.session_state.y_test is None:
        st.warning("Train the model first in the Model Training step.")
    else:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        y_true = st.session_state.y_test
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("Accuracy", f"{accuracy:.2f}")
        metrics_col2.metric("Precision", f"{precision:.2f}")
        metrics_col3.metric("Recall", f"{recall:.2f}")
        metrics_col4.metric("F1-score", f"{f1:.2f}")

        st.markdown("---")
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        st.pyplot(fig_cm)

        st.markdown("---")
        st.subheader("ROC Curve")
        y_score = st.session_state.model.predict_proba(st.session_state.X_test)
        classes = np.unique(y_true)
        fig_roc = plot_roc_curve(y_true, y_score, classes)
        st.pyplot(fig_roc)

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
