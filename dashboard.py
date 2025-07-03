import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

from utils import load_data, load_data_from_upload
from ml_models import (
    kmeans_cluster, get_elbow_curve, get_silhouette_scores, plot_elbow_curve, plot_silhouette_curve,
    cluster_3d_plot, train_classifiers, plot_confusion, train_regressors
)

st.set_page_config(page_title="CyberSOC-aaS Executive Dashboard", layout="wide")
st.title("Blockchain-Enabled Agentic AI Cybersecurity Platform")

# ---------------- DATA LOAD LOGIC ----------------
data_mode = st.sidebar.radio("Data Source", ["Backend CSVs", "Upload CSVs"], index=0)
dfs = {}
if data_mode == "Backend CSVs":
    dfs = load_data('data')
    missing_files = [k for k, v in dfs.items() if v is None]
    if missing_files:
        st.sidebar.warning(f"Missing: {', '.join(missing_files)} in /data/. Please upload them or switch mode.")
        st.stop()
else:
    st.sidebar.header("Upload All 9 CSV Data Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload all: employees, departments, roles, training, incidents, security_events, agentic_ai_log, blockchain_audit_log, policy",
        type="csv", accept_multiple_files=True)
    if not uploaded_files or len(uploaded_files) < 9:
        st.sidebar.warning("Please upload all 9 required CSV files.")
        st.stop()
    dfs = load_data_from_upload(uploaded_files)

# Unpack all tables
df_emp = dfs['employees']
df_dept = dfs['departments']
df_roles = dfs['roles']
df_training = dfs['training']
df_incidents = dfs['incidents']
df_events = dfs['security_events']
df_ai = dfs['agentic_ai_log']
df_bc = dfs['blockchain_audit_log']
df_policy = dfs['policy']

# ---------------- TABS ----------------
tabs = st.tabs([
    "🏠 Executive Overview",
    "📊 Cybersecurity Metrics",
    "🤖 Agentic AI & Blockchain",
    "🧠 ML & Predictive Analytics",
    "🕵️ Employee/Asset Explorer"
])

# ---------------- EXECUTIVE OVERVIEW ----------------
with tabs[0]:
    st.header("Executive Summary & KPIs")
    st.markdown("""
    **Business Impact:**  
    This dashboard gives the C-suite a real-time pulse on cyber risk, training, and readiness.  
    _Use for board meetings, strategy reviews, and compliance reporting._
    """)

    # Top risk department and risk banner
    dept_risk = df_incidents.groupby('DepartmentID').size().sort_values(ascending=False)
    risk_dept = df_dept[df_dept['DepartmentID'] == dept_risk.index[0]]['DepartmentName'].values[0]
    critical_inc = df_incidents[df_incidents['Severity'] == 'Critical']
    if not critical_inc.empty:
        top_crit_dept = df_dept[df_dept['DepartmentID'] == critical_inc['DepartmentID'].value_counts().idxmax()]['DepartmentName'].values[0]
        st.error(f"🚨 Highest critical incidents in {top_crit_dept} department! Investigate root cause.")
    else:
        st.success("No critical incidents reported in last month. Keep up the good work!")
    st.info(f"**AI Recommendation:** Focus awareness training on {risk_dept} department (highest incident rate).")

    # Key business metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", df_emp.shape[0])
    col2.metric("Total Incidents (yr)", df_incidents.shape[0])
    col3.metric("Trainings Completed", (df_training['Status'] == "Completed").sum())
    col4.metric("Resolved Incidents", (df_incidents['Status'] == "Closed").sum())

    st.subheader("Workforce Cyber Awareness")
    st.write("The distribution of cybersecurity awareness scores (should be right-skewed if staff are well-trained).")
    awareness = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness']
    fig_awareness = px.histogram(awareness, x="Score", nbins=25, title="Employee Awareness Scores", color="Status")
    st.plotly_chart(fig_awareness, use_container_width=True)

    st.subheader("Incident Trends")
    st.write("Incidents reported each month. Look for spikes (threat waves, process changes, employee churn).")
    df_incidents['DateReported'] = pd.to_datetime(df_incidents['DateReported'])
    trend = df_incidents.groupby(df_incidents['DateReported'].dt.to_period('M')).size()
    fig_trend = px.line(
        x=trend.index.astype(str), y=trend.values, markers=True,
        title="Monthly Security Incident Reports", labels={'x': 'Month', 'y': 'Number of Incidents'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Active Compliance Policies")
    st.write("Security policies with versioning and AI/Human enforcement. Useful for audits.")
    st.dataframe(df_policy.sort_values('DateActive', ascending=False).head(8), use_container_width=True)

# ---------------- CYBERSECURITY METRICS ----------------
with tabs[1]:
    st.header("Operational Cybersecurity Metrics")
    st.markdown("""
    **Business Impact:**  
    Use these for risk management, resource allocation, and department reviews.
    """)
    st.subheader("Incident Type Distribution")
    fig_types = px.pie(df_incidents, names="IncidentType", title="Incidents by Type")
    st.plotly_chart(fig_types, use_container_width=True)

    st.subheader("Incident Severity by Department")
    dept_sev = pd.crosstab(df_incidents['DepartmentID'], df_incidents['Severity'])
    dept_sev.index = dept_sev.index.map(df_dept.set_index('DepartmentID')['DepartmentName'].to_dict())
    fig_heat = px.imshow(dept_sev, text_auto=True, aspect="auto", title="Severity by Department")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Top Risk Departments")
    top_depts = df_incidents['DepartmentID'].value_counts().rename(index=df_dept.set_index('DepartmentID')['DepartmentName'].to_dict())
    fig_bar = px.bar(x=top_depts.index, y=top_depts.values, labels={'x': 'Department', 'y': 'Incident Count'}, title="Incident Volume by Department")
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- AGENTIC AI & BLOCKCHAIN ----------------
with tabs[2]:
    st.header("Agentic AI Automation & Blockchain Audit")
    st.subheader("Latest Agentic AI Actions")
    st.dataframe(
        df_ai[['Timestamp','AgentID','ActionTaken','DecisionConfidence','EscalatedToHuman','Outcome']]
        .sort_values('Timestamp', ascending=False).head(30), use_container_width=True
    )
    st.subheader("AI Decision Confidence by Action")
    fig_conf = px.box(df_ai, x='ActionTaken', y='DecisionConfidence', title="AI Decision Confidence by Action")
    st.plotly_chart(fig_conf, use_container_width=True)
    st.subheader("Blockchain Audit Log")
    st.dataframe(df_bc.sample(10), use_container_width=True)

# ---------------- EMPLOYEE/ASSET EXPLORER ----------------
with tabs[4]:
    st.header("Employee/Asset Risk Explorer")
    st.markdown("""
    Drill into individual profiles for cyber readiness, incident history, and training status.
    """)
    emp_name = st.selectbox("Select Employee", df_emp['Name'])
    emp_row = df_emp[df_emp['Name'] == emp_name].iloc[0]
    st.write("### Profile Summary")
    st.json(emp_row.to_dict())
    st.write("#### Trainings Completed")
    st.dataframe(df_training[df_training['EmployeeID'] == emp_row['EmployeeID']], use_container_width=True)
    st.write("#### Incident History")
    st.dataframe(df_incidents[df_incidents['EmployeeID'] == emp_row['EmployeeID']], use_container_width=True)
    st.write("#### Security Events (sample)")
    st.dataframe(df_events[df_events['EmployeeID'] == emp_row['EmployeeID']].head(10), use_container_width=True)

# ---------------- ML & PREDICTIVE ANALYTICS (With Bulletproof 3D) ----------------
with tabs[3]:
    st.header("ML & Predictive Analytics")
    st.markdown("""
    Boardroom-ready risk analytics.  
    - 3D clustering: Find risk clusters and outliers among staff  
    - Classification: Predict incident severity, tune response  
    - Regression: See what factors drive incident risk
    """)

    # -- Feature Engineering (always include needed columns) --
    df_emp_ml = df_emp[['EmployeeID', 'Name', 'DepartmentID', 'RoleID']].copy()
    emp_awareness = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness'].groupby('EmployeeID')['Score'].mean().reset_index()
    emp_incidents = df_incidents.groupby('EmployeeID').size().reset_index(name='IncidentsCaused')
    emp_training_count = df_training.groupby('EmployeeID').size().reset_index(name='TrainingCount')
    df_emp_ml = df_emp_ml.merge(emp_awareness, on='EmployeeID', how='left')
    df_emp_ml = df_emp_ml.merge(emp_incidents, on='EmployeeID', how='left').fillna({'Score':0, 'IncidentsCaused':0})
    df_emp_ml = df_emp_ml.merge(emp_training_count, on='EmployeeID', how='left').fillna({'TrainingCount':0})

    st.subheader("3D Risk Clustering")
    cluster_features = st.multiselect("Clustering Features", ['Score', 'IncidentsCaused', 'TrainingCount'], default=['Score','IncidentsCaused','TrainingCount'])
    max_k = st.slider("Elbow/Silhouette: Max clusters (K)", min_value=3, max_value=10, value=5)
    if len(cluster_features) >= 3:
        X = df_emp_ml[cluster_features].dropna().to_numpy()
        inertia = get_elbow_curve(X, max_k=max_k)
        sil_scores = get_silhouette_scores(X, max_k=max_k)
        st.plotly_chart(plot_elbow_curve(inertia, max_k=max_k), use_container_width=True)
        st.plotly_chart(plot_silhouette_curve(sil_scores, max_k=max_k), use_container_width=True)
        n_clusters = st.slider("Number of Clusters to Display", min_value=2, max_value=max_k, value=3)
        df_clustered, kmeans, sil_score = kmeans_cluster(df_emp_ml, cluster_features, n_clusters)
        # Ensure custom_data columns exist for 3D
        for col in ['EmployeeID', 'Name', 'DepartmentID', 'RoleID']:
            if col not in df_clustered.columns:
                df_clustered[col] = None
        st.success(f"Silhouette Score for {n_clusters} clusters: {sil_score:.3f}")
        fig3d = cluster_3d_plot(df_clustered, cluster_features[0], cluster_features[1], cluster_features[2])
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("Click a 3D point for employee drilldown info (hover shows details).")

    st.divider()
    st.subheader("Incident Severity Classification (All Models)")
    df_inc = df_incidents.copy()
    emp_feats = df_emp[['EmployeeID', 'DepartmentID', 'RoleID', 'Location']]
    train_feats = df_training[df_training['TrainingType']=='Cybersecurity Awareness'].groupby('EmployeeID')['Score'].mean().reset_index()
    df_inc = df_inc.merge(emp_feats, left_on='EmployeeID', right_on='EmployeeID', how='left')
    df_inc = df_inc.merge(train_feats, on='EmployeeID', how='left', suffixes=('','_awareness'))
    df_inc['DeptIdx'] = df_inc['DepartmentID'].astype('category').cat.codes
    df_inc['RoleIdx'] = df_inc['RoleID'].astype('category').cat.codes
    df_inc['LocationIdx'] = df_inc['Location'].astype('category').cat.codes
    df_inc['TypeIdx'] = df_inc['IncidentType'].astype('category').cat.codes
    X_cols = ['AIResponseTime', 'DeptIdx', 'RoleIdx', 'LocationIdx', 'TypeIdx', 'Score']
    df_inc = df_inc.dropna(subset=X_cols + ['Severity'])
    label_encoder = LabelEncoder()
    if df_inc.empty:
        st.warning("Not enough clean data for classification. Please check for missing values.")
    else:
        models, y_test, label_encoder = train_classifiers(df_inc, X_cols, 'Severity', label_encoder=label_encoder)
        for name, (model, score, cm, *extras) in models.items():
            st.write(f"**{name}**: Test accuracy = {score:.2%}")
            st.plotly_chart(plot_confusion(cm, labels=label_encoder.classes_, title=f"{name} Confusion Matrix"), use_container_width=True)
            if name == "RF" and extras:
                st.write("**Random Forest Feature Importances**")
                st.bar_chart(pd.Series(extras[0], index=X_cols))
        st.write("**Interpretation:** Focus on models with highest Critical/High class accuracy and on features with highest importance.")

    st.divider()
    st.subheader("Incident Cause Risk Regression (All Models)")
    df_reg = df_emp_ml.copy()
    df_reg['RoleIdx'] = df_reg['RoleID']
    regression_features = st.multiselect("Regression Features", ['Score', 'RoleIdx', 'TrainingCount'], default=['Score','RoleIdx','TrainingCount'])
    target_reg = 'IncidentsCaused'
    if df_reg[regression_features + [target_reg]].dropna().empty:
        st.warning("Not enough clean data for regression. Please select valid features.")
    else:
        regressors = train_regressors(df_reg, regression_features, target_reg)
        for name, (model, r2, coef) in regressors.items():
            st.write(f"**{name} Regression**: R2 = {r2:.2f}")
            if coef is not None:
                st.bar_chart(pd.Series(coef, index=regression_features))
        st.write("**Interpretation:** Features with largest coefficients are most important for boardroom risk planning.")

    st.info("""
    _This analytics panel transforms AI/ML insights into CxO-level decisions:  
    - Who's most at risk?  
    - Where to invest?  
    - How to lower breach odds next quarter?_
    """)
