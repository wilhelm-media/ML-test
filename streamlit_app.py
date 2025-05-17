import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(layout="wide")

st.title("Absprungs-Raten Analyse & Vorhersage")

# 📁 Datei-Upload
uploaded_file = st.file_uploader("CSV-Datei mit Kundendaten hochladen", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1️⃣ Vorschau der Daten")
    st.dataframe(df.head())

    # 🧼 Preprocessing
    df_encoded = df.copy()
    le = LabelEncoder()
    if 'contract_type' in df_encoded.columns:
        df_encoded['contract_type'] = le.fit_transform(df_encoded['contract_type'])

    # Spalten prüfen
    if "churn" not in df.columns:
        st.warning("⚠️ Spalte 'churn' fehlt – bitte sicherstellen, dass deine Daten korrekt sind.")
        st.stop()

    # Modell-Training
    X = df_encoded.drop(columns=["customer_id", "churn"])
    y = df_encoded["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 🔍 Visuelle Analyse
    st.subheader("2️⃣ Datenvisualisierung")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Verteilung nach Vertragsart:**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x="contract_type", hue="churn", ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("**Monatliche Ausgaben (Churn vs. No Churn):**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x="churn", y="monthly_spend", ax=ax2)
        st.pyplot(fig2)

    # 🔮 Churn Prediction für neuen Kunden
    st.subheader("3️⃣ Churn-Vorhersage für neuen Kunden")

    def user_input_features():
        tenure = st.slider("Kundentreue in Monaten", 0, 72, 12)
        spend = st.slider("Monatlicher Umsatz (€)", 0, 500, 50)
        logins = st.slider("Login-Häufigkeit (Monat)", 0, 50, 10)
        contract = st.selectbox("Vertragstyp", df['contract_type'].unique())

        contract_encoded = le.transform([contract])[0] if 'contract_type' in df.columns else 0
        data = {
            "tenure_months": tenure,
            "monthly_spend": spend,
            "login_frequency": logins,
            "contract_type": contract_encoded
        }
        return pd.DataFrame([data])

    input_df = user_input_features()

    st.markdown("**Benutzereingaben:**")
    st.write(input_df)

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.markdown("### 🔔 Ergebnis:")
    st.success(f"Churn-Wahrscheinlichkeit: **{prediction_proba:.2%}**")
    st.info("Warnung: Hoch = Kunde könnte abspringen." if prediction else "✅ Kunde ist stabil.")

    # 📈 Feature Importance
    st.subheader("4️⃣ Wichtigste Einflussfaktoren")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    feature_names = X.columns[sorted_idx]

    fig3, ax3 = plt.subplots()
    ax3.barh(range(len(feature_names)), importances[sorted_idx])
    ax3.set_yticks(range(len(feature_names)))
    ax3.set_yticklabels(feature_names)
    ax3.set_title("Feature Importance")
    st.pyplot(fig3)
