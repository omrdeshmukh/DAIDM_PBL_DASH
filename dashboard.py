import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data_from_upload
from ml_models import kmeans_cluster, cluster_plot, severity_classifier, regression_analysis

st.set_page_config(page_title="CyberSOC-aaS Executive Dashboard", layout="wide")
st.title("Blockchain-Enabled Agentic AI Cybersecurity Platform")

# ---- Upload UI ----
st.sidebar.header("Upload All 9 CSV Data Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload the following: employees, departments, roles, training, incidents, security_events, agentic_ai_log, blockchain_audit_log, policy",
    type="csv", accept_multiple_files=True)

if not uploaded_files or len(uploaded_files) < 9:
    st.warning("Please upload all 9 required CSV files.")
    st.stop()

dfs = load_data_from_upload(uploaded_files)

# Assign DataFrames
try:
    df_emp = dfs['employees']
    df_dept = dfs['departments']
    df_roles = dfs['roles']
    df_training = dfs['training']
    df_incidents = dfs['incidents']
    df_events = dfs['security_events']
    df_ai = dfs['agentic_ai_log']
    df_bc = dfs['blockchain_audit_log']
    df_policy = dfs['policy']
except Exception as e:
    st.error(f"Could not assign all dataframes: {e}")
    st.stop()

# ---- Tabs ----
tabs = st.tabs([
    "🏠 Executive Overview", 
    "📊 Cybersecurity Metrics", 
    "🔍 Incident & Threat Analytics", 
    "🤖 Agentic AI & Blockchain", 
    "🧠 ML & Predictive Analytics", 
    "🕵️ Employee/Asset Explorer"
])

# --- EXECUTIVE OVERVIEW ---
with tabs[0]:
    st.header("Executive Summary & KPIs")
    st.write("A high-level summary for C-suite, including real-time security KPIs, trends, and compliance posture.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", df_emp.shape[0])
    col2.metric("Total Incidents (year)", df_incidents.shape[0])
    col3.metric("Trainings (completed)", (df_training['Status']=="Completed").sum())
    col4.metric("Resolved Incidents", (df_incidents['Status']=="Closed").sum())

    st.subheader("Training & Awareness Completion")
    st.write("**This chart shows the distribution of cybersecurity awareness scores across all employees. Peaks to the right indicate successful training.**")
    awareness = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness']
    fig_awareness = px.histogram(awareness, x="Score", nbins=25, title="Awareness Score Distribution", color="Status")
    st.plotly_chart(fig_awareness, use_container_width=True)

    st.subheader("Incident Trend (Last Year)")
    st.write("**Tracks the volume of reported incidents each month. Spikes may correlate to attack waves or new vulnerabilities.**")
    df_incidents['DateReported'] = pd.to_datetime(df_incidents['DateReported'])
    monthly_trend = df_incidents.groupby(df_incidents['DateReported'].dt.to_period('M')).size()
    fig_trend = px.line(x=monthly_trend.index.astype(str), y=monthly_trend.values, markers=True,
                        title="Incidents Reported Over Time", labels={'x': 'Month', 'y': 'Incident Count'})
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Compliance Policy Versions")
    st.write("**Recent active security policies and enforcement records (AI/Human/Hybrid).**")
    st.dataframe(df_policy.sort_values('DateActive', ascending=False).head(8), use_container_width=True)

# --- CYBERSECURITY METRICS ---
with tabs[1]:
    st.header("Operational Security Metrics")
    st.write("**Monitor incident types, severities, department risks, and threat heatmaps to prioritize resource allocation and risk mitigation.**")
    fig_types = px.pie(df_incidents, names="IncidentType", title="Incidents by Type")
    st.plotly_chart(fig_types, use_container_width=True)
    dept_sev = pd.crosstab(df_incidents['DepartmentID'], df_incidents['Severity'])
    dept_sev.index = dept_sev.index.map(df_dept.set_index('DepartmentID')['DepartmentName'].to_dict())
    fig_heat = px.imshow(dept_sev, text_auto=True, aspect="auto", title="Incident Severity by Department")
    st.plotly_chart(fig_heat, use_container_width=True)
    top_depts = df_incidents['DepartmentID'].value_counts().rename(index=df_dept.set_index('DepartmentID')['DepartmentName'].to_dict())
    fig_bar = px.bar(x=top_depts.index, y=top_depts.values, labels={'x':'Department','y':'Incident Count'}, title="Incidents by Department")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- INCIDENT & THREAT ANALYTICS ---
with tabs[2]:
    st.header("Incident & Threat Deep-Dive")
    st.write("**Analyze individual incidents, threat detection speed, AI vs. human response, and event-level logs.**")
    sev_filter = st.multiselect("Filter by Severity", df_incidents['Severity'].unique(), default=df_incidents['Severity'].unique())
    type_filter = st.multiselect("Filter by Type", df_incidents['IncidentType'].unique(), default=df_incidents['IncidentType'].unique())
    sub_df = df_incidents[df_incidents['Severity'].isin(sev_filter) & df_incidents['IncidentType'].isin(type_filter)]
    st.dataframe(sub_df[['IncidentID','IncidentType','Severity','DateReported','Status','AgenticAIAction','AIResponseTime']].sort_values('DateReported', ascending=False).head(30), use_container_width=True)
    st.write("**Scatterplot below shows AI response time vs. incident severity.**")
    fig_resp = px.scatter(sub_df, x="AIResponseTime", y="Severity", color="Status", 
                          title="AI Response Time vs. Severity", labels={"AIResponseTime":"AI Response Time (min)"})
    st.plotly_chart(fig_resp, use_container_width=True)

# --- AGENTIC AI & BLOCKCHAIN ---
with tabs[3]:
    st.header("Agentic AI Automation & Blockchain Auditability")
    st.write("**Explore logs of automated actions, AI confidence, human escalations, and blockchain-backed event trails.**")
    st.write("Agentic AI Action Log (last 20):")
    st.dataframe(df_ai[['Timestamp','AgentID','ActionTaken','DecisionConfidence','EscalatedToHuman','Outcome']].sort_values('Timestamp', ascending=False).head(20), use_container_width=True)
    fig_conf = px.box(df_ai, x='ActionTaken', y='DecisionConfidence', title="AI Decision Confidence by Action")
    st.plotly_chart(fig_conf, use_container_width=True)
    st.write("Blockchain Audit Log (random sample):")
    st.dataframe(df_bc.sample(10), use_container_width=True)

# --- ML & PREDICTIVE ANALYTICS ---
with tabs[4]:
    st.header("Machine Learning Analytics")
    st.write("**Leverage unsupervised and supervised ML to uncover security clusters, predict incident severity, and analyze risk factors.**")
    st.subheader("K-Means Clustering & Silhouette")
    st.write("**Cluster employees by average awareness score and number of incidents caused.**")
    df_emp_ml = df_emp.copy()
    emp_awareness = df_training[df_training['TrainingType']=='Cybersecurity Awareness'].groupby('EmployeeID')['Score'].mean().reset_index()
    emp_incidents = df_incidents.groupby('EmployeeID').size().reset_index(name='IncidentsCaused')
    df_emp_ml = pd.merge(df_emp_ml, emp_awareness, on='EmployeeID', how='left')
    df_emp_ml = pd.merge(df_emp_ml, emp_incidents, on='EmployeeID', how='left').fillna({'Score':0, 'IncidentsCaused':0})
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=3)
    df_clustered, kmeans, sil_score = kmeans_cluster(df_emp_ml, ['Score', 'IncidentsCaused'], n_clusters)
    st.info(f"Silhouette Score: {sil_score:.3f} (higher is better cluster separation)")
    st.plotly_chart(cluster_plot(df_clustered, 'Score', 'IncidentsCaused'), use_container_width=True)
    st.write("**Interpretation:** Each color represents a cluster of employees based on cyber awareness and incident history. Outliers/high-risk employees stand apart.")

    st.subheader("Incident Severity Classification (Random Forest)")
    st.write("**Predict incident severity based on features (AI response time, department, incident type).**")
    feat_map = {k: v for v, k in enumerate(df_dept['DepartmentName'].values)}
    df_class = df_incidents.copy()
    df_class['DeptIdx'] = df_class['DepartmentID'].map(feat_map)
    df_class['TypeIdx'] = df_class['IncidentType'].astype('category').cat.codes
    
    # Ensure all features and target are numeric and drop rows with missing or infinite values
    X_cols = ['AIResponseTime', 'DeptIdx', 'TypeIdx']
    for col in X_cols:
        df_class[col] = pd.to_numeric(df_class[col], errors='coerce')
    df_class = df_class.dropna(subset=X_cols + ['Severity'])
    
    if df_class.empty:
        st.warning("Not enough clean data for classification. Please check for missing AIResponseTime or Severity values.")
    else:
        if df_class.empty:
            st.warning("Not enough clean data for classification. Please check for missing AIResponseTime or Severity values.")
        else:
            try:
                model, importances = severity_classifier(df_class, X_cols, target='Severity')
                st.bar_chart(pd.Series(importances, index=X_cols))
                st.write("**Interpretation:** Higher feature importance means that factor is more predictive of incident severity. Use these insights to tune training and response.")
            except Exception as e:
                st.error(f"Classification model failed: {e}")
                
    st.write("**Interpretation:** Higher feature importance means that factor is more predictive of incident severity. Use these insights to tune training and response.")

    st.subheader("Regression: Predicting Number of Incidents Caused by Employee")
    st.write("**Linear regression predicts who might cause more incidents (risk scoring) based on training scores and role.**")
    df_reg = df_emp_ml.copy()
    df_reg['RoleIdx'] = df_reg['RoleID']
    features = ['Score', 'RoleIdx']
    model_reg, coefs = regression_analysis(df_reg, features, 'IncidentsCaused')
    st.write("**Feature coefficients:**")
    st.write(pd.Series(coefs, index=features))
    st.write("**Interpretation:** A higher coefficient means that factor has a bigger effect on the likelihood of causing incidents.")

# --- EMPLOYEE/ASSET EXPLORER ---
with tabs[5]:
    st.header("Employee or Asset Profile Explorer")
    st.write("**Drill down into any employee or device to see training, incidents, and risk profile.**")
    emp_name = st.selectbox("Select Employee", df_emp['Name'])
    emp_row = df_emp[df_emp['Name'] == emp_name].iloc[0]
    st.write("**Profile**")
    st.write(emp_row)
    st.write("**Trainings**")
    st.dataframe(df_training[df_training['EmployeeID'] == emp_row['EmployeeID']], use_container_width=True)
    st.write("**Incidents**")
    st.dataframe(df_incidents[df_incidents['EmployeeID'] == emp_row['EmployeeID']], use_container_width=True)
    st.write("**Security Events**")
    st.dataframe(df_events[df_events['EmployeeID'] == emp_row['EmployeeID']].head(10), use_container_width=True)

st.info("This dashboard empowers CEO & CTO with actionable insights on cyber readiness, incident risk, AI efficacy, and compliance. Use tabs to explore metrics, drill into root causes, and optimize security posture.")
