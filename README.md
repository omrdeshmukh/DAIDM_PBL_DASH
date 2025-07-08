# 🛡️ CyberSOC-aaS: Agentic AI & Blockchain-Powered Cybersecurity Dashboard

This Streamlit dashboard is part of a cybersecurity analytics platform that uses AI, blockchain, and machine learning to assess, predict, and reduce organizational cyber risks—especially those tied to human factors like awareness, training, and insider actions.

🔗 **Live Demo:** [daidmpbldash.streamlit.app](https://daidmpbldash.streamlit.app)

## 🔍 Overview

This platform is built to serve executive stakeholders—CEOs, CTOs, CISOs, and data teams—offering them real-time insights into:

- Cybersecurity awareness and training progress
- Incident trends and severity breakdowns
- Predictive modeling of employee risk
- Persona-based risk clusters
- Blockchain audit trails of AI agent actions
- Compliance and policy enforcement patterns

## 📊 Key Features

### ✅ Executive Summary KPIs
- Training completion rates
- Average awareness score
- Total and resolved incidents (last 12 months)

### 🔐 Cybersecurity Incident Trends
- Incidents by type, department, hour, and severity
- Lag in reporting vs resolution
- Internal vs external incident breakdown

### 🤖 Agentic AI & Blockchain Activity
- Log of AI interventions (e.g., auto-lockouts, escalations)
- Confidence scores of AI decisions
- Tamper-proof blockchain audit logs

### 🧠 Predictive Analytics & Machine Learning
- Classification of incident severity (KNN, DT, RF, GBRT)
- Risk scoring with Random Forest (94% accuracy)
- Clustering of employee personas (e.g., "At-Risk Newcomers")
- Feature importance and model comparison charts
- Regression to quantify impact of training/awareness on incidents

### 📈 Association Rule Mining
- Discovered logic like:
  - _“Untrained Sales employees → 80% chance of phishing failure”_
  - _“Low awareness + staff role → high chance of incident”_

### 👥 Role & Persona Explorer
- Select employee to view training, incident, and event history
- Filtered views for HR, IT, or executive stakeholders

---

## 📁 File Structure

```bash
📦 CyberSOC-aaS/
├── dashboard.py              # Streamlit app entry point
├── ml_models.py              # All ML training, clustering, regression code
├── utils.py                  # Data loading, preprocessing helpers
├── data/
│   ├── employees.csv
│   ├── incidents.csv
│   └── ... (9 total datasets)
├── requirements.txt          # Python dependencies
├── README.md                 # You're reading it!
````

---

## 🚀 How to Run Locally

1. Clone this repo:

   ```bash
   git clone https://github.com/yourusername/cybersoc-dashboard.git
   cd cybersoc-dashboard
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch dashboard:

   ```bash
   streamlit run dashboard.py
   ```

4. Upload all 9 CSV files when prompted, or place them in the `data/` folder.

---

## 🧠 Algorithms Used

* Classification: K-Nearest Neighbors, Decision Trees, Random Forest, Gradient Boosted Trees
* Clustering: K-Means (with Elbow & Silhouette analysis)
* Regression: Linear, Ridge, and Lasso
* Association Rule Mining: Apriori Algorithm

---

## 🎯 Use Cases

* **CxOs** can monitor security posture, training compliance, and threats
* **IT & SOC teams** can identify high-risk users and act early
* **HR teams** can evaluate awareness levels and tailor onboarding
* **Auditors/Regulators** get transparency via blockchain logs

---

## 📌 Live Demo (Public View)

👉 [https://daidmpbldash.streamlit.app](https://daidmpbldash.streamlit.app)

Feel free to explore all 10+ tabs covering 100+ KPIs, predictive models, and live ML visualizations.

---

## 📬 Contact

This dashboard is developed as part of an MBA capstone/industry-facing project.

For collaboration, queries, or deployment help:
📧 [om.rdeshmukh3@gmail.com](mailto:om.rdeshmukh3@gmail.com)
📍 United Arab Emirates

---

## ⭐ Future Enhancements

* Add login/user role management
* Schedule daily/weekly data refresh
* Integration with real-time alerts (e.g., Slack, Email)
* Expand to cloud-native data sources (BigQuery, Snowflake)

---

## 📜 License

This project is open for academic, educational, and demonstrative purposes. Contact before commercial use.

