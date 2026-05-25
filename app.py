"""
Simple Streamlit demo for the Bank Telemarketing DSS.
"""

import pickle

import numpy as np
import streamlit as st  # type: ignore[reportMissingImports]


st.set_page_config(page_title="Bank Telemarketing DSS", layout="wide")


#Δεν χρησιμοποιώ caching, ώστε να διαβάζω πάντα τα πιο πρόσφατα artifacts του pipeline.
try:
    with open("outputs/artifacts.pkl", "rb") as f:
        art = pickle.load(f)
except FileNotFoundError:
    st.error("Run python run_all.py first, so that outputs/artifacts.pkl is created.")
    st.stop()


cluster_names = {
    0: "Blue-Collar",
    1: "Mainstream",
    2: "Unknown Status",
    3: "Prior Success",
}


st.title("Bank Telemarketing DSS")
st.caption("Demo εφαρμογή για πρόβλεψη ανταπόκρισης πελάτη σε τραπεζική καμπάνια.")

left, right = st.columns(2)

with left:
    st.subheader("Στοιχεία πελάτη")
    age = st.number_input("Ηλικία", min_value=18, max_value=100, value=40)
    job = st.selectbox(
        "Επάγγελμα",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ],
    )
    marital = st.selectbox("Οικογενειακή κατάσταση", ["married", "single", "divorced", "unknown"])
    education = st.selectbox(
        "Εκπαίδευση",
        [
            "university.degree",
            "high.school",
            "basic.9y",
            "professional.course",
            "basic.4y",
            "basic.6y",
            "illiterate",
            "unknown",
        ],
    )
    housing = st.selectbox("Στεγαστικό δάνειο", ["yes", "no", "unknown"])
    loan = st.selectbox("Προσωπικό δάνειο", ["no", "yes", "unknown"])

with right:
    st.subheader("Στοιχεία καμπάνιας")
    contact = st.selectbox("Τύπος επικοινωνίας", ["cellular", "telephone"])
    month = st.selectbox("Μήνας", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    day_of_week = st.selectbox("Ημέρα", ["mon", "tue", "wed", "thu", "fri"])
    campaign = st.number_input("Επαφές τρέχουσας καμπάνιας", min_value=1, max_value=50, value=1)
    pdays = st.number_input("Ημέρες από προηγούμενη επαφή", min_value=0, max_value=999, value=999)
    previous = st.number_input("Προηγούμενες επαφές", min_value=0, max_value=20, value=0)
    poutcome = st.selectbox("Αποτέλεσμα προηγούμενης καμπάνιας", ["nonexistent", "failure", "success"])

st.subheader("Μακροοικονομικοί δείκτες")
macro_cols = st.columns(5)
with macro_cols[0]:
    emp_var_rate = st.number_input("emp.var.rate", value=1.1)
with macro_cols[1]:
    cons_price_idx = st.number_input("cons.price.idx", value=93.994)
with macro_cols[2]:
    cons_conf_idx = st.number_input("cons.conf.idx", value=-36.4)
with macro_cols[3]:
    euribor3m = st.number_input("euribor3m", value=4.857)
with macro_cols[4]:
    nr_employed = st.number_input("nr.employed", value=5191.0)


if st.button("Πρόβλεψη", type="primary"):
    customer = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": "no",
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }

    feature_names = art["feature_names"]
    num_cols = art["num_cols"]
    cat_cols = art["cat_cols"]
    edu_order = art["edu_order"]

    #Το education κωδικοποιείται ακριβώς όπως στο preprocessing για πλήρη συνέπεια.
    edu_map = {}
    for i in range(len(edu_order)):
        edu_map[edu_order[i]] = i
    edu_numeric = edu_map.get(customer.get("education", "unknown"), len(edu_order) - 1)

    #Χτίζω το row με τη σειρά του feature_names, όπως εκπαιδεύτηκε το μοντέλο.
    row = np.zeros(len(feature_names))
    for i in range(len(feature_names)):
        feature = feature_names[i]
        if feature in num_cols:
            if feature == "education":
                row[i] = edu_numeric
            else:
                row[i] = customer.get(feature, 0)
        else:
            #Αντιστοίχιση one-hot με βάση το πρόθεμα της αρχικής κατηγορικής στήλης.
            for col in cat_cols:
                prefix = col + "_"
                if feature.startswith(prefix):
                    value = feature[len(prefix):]
                    if customer.get(col) == value:
                        row[i] = 1.0
                    else:
                        row[i] = 0.0
                    break

    X = row.reshape(1, -1)
    X_scaled = art["scaler"].transform(X)

    #Το τελικό μοντέλο διαβάζεται από τα artifacts, χωρίς hardcoded επιλογή.
    model_name = art.get("decision_model", "XGB")
    model = art["models"][model_name]
    probability = model.predict_proba(X_scaled)[0, 1]

    #Πρώτα υπολογίζω το cluster και μετά εφαρμόζω το αντίστοιχο threshold.
    cluster_input = X_scaled[:, art["cluster_feature_idx"]]
    cluster = int(art["kmeans"].predict(cluster_input)[0])
    threshold = art["cluster_thresholds"][cluster]
    decision = probability >= threshold

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Μοντέλο", model_name)
    c2.metric("Πιθανότητα", f"{probability:.1%}")
    c3.metric("Cluster", f"C{cluster}: {cluster_names.get(cluster, 'Unknown')}")
    c4.metric("Threshold", f"{threshold:.0%}")

    if decision:
        st.success("Απόφαση: ΚΑΛΕΣΕ τον πελάτη.")
    else:
        st.warning("Απόφαση: ΜΗΝ καλέσεις τον πελάτη.")
