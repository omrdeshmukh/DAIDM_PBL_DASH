import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_data

# ---- Load Data ----
df_emp, df_dept, df_roles, df_training, df_incidents, df_events, df_ai, df_bc, df_policy = load_data()

st.set_page_config(page_title="CyberSOC-aaS Dashboard", layout="wide")
st.title("Blockchain-Enabled Agentic AI Cybersecurity Platform Dashboard")

# ---- SIDEBAR FILTERS ----
with st.sidebar:
    st.header("Filters")
    dept_opt = st.multiselect("Department", df_dept['DepartmentName'].tolist())
    loc_opt = st.multiselect("Location", sorted(df_emp['Location'].unique()))
    role_opt = st.multiselect("Role", df_roles['RoleName'].tolist())
    # Date Range Filter for incidents/events
    min_dt, max_dt = pd.to_datetime(df_incidents['DateReported']).min(), pd.to_datetime(df_incidents['DateReported']).max()
    date_rng = st.date_input("Incident Date Range", [min_dt, max_dt])

# ---- APPLY FILTERS ----
filtered_emp = df_emp.copy()
if dept_opt:
    dept_ids = df_dept[df_dept['DepartmentName'].isin(dept_opt)]['DepartmentID']
    filtered_emp = filtered_emp[filtered_emp['DepartmentID'].isin(dept_ids)]
if loc_opt:
    filtered_emp = filtered_emp[filtered_emp['Location'].isin(loc_opt)]
if role_opt:
    role_ids = df_roles[df_roles['RoleName'].isin(role_opt)]['RoleID']
    filtered_emp = filtered_emp[filtered_emp['RoleID'].isin(role_ids)]

# Filter incidents by selected employees and date
filtered_incidents = df_incidents[df_incidents['EmployeeID'].isin(filtered_emp['EmployeeID'])]
filtered_incidents = filtered_incidents[
    (pd.to_datetime(filtered_incidents['DateReported']) >= pd.to_datetime(date_rng[0])) &
    (pd.to_datetime(filtered_incidents['DateReported']) <= pd.to_datetime(date_rng[1]))
]

# ---- DASHBOARD METRICS ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", len(filtered_emp))
col2.metric("Incidents Reported", len(filtered_incidents))
trainings_completed = df_training[(df_training['EmployeeID'].isin(filtered_emp['EmployeeID'])) & (df_training['Status']=="Completed")]
col3.metric("Trainings Completed", len(trainings_completed))
col4.metric("Resolved Incidents", filtered_incidents['Status'].value_counts().get('Closed', 0))

# ---- TRAINING & AWARENESS ----
st.subheader("Training & Awareness")
# Training completion rate per department
df_emp_training = pd.merge(df_emp, df_training[df_training['Status']=='Completed'], on='EmployeeID', how='left')
train_rate = df_emp_training.groupby('DepartmentID').agg(
    Emp_Count=('EmployeeID', 'nunique'),
    Trainings_Completed=('TrainingID', 'count')
).reset_index()
train_rate = pd.merge(train_rate, df_dept, on='DepartmentID')

fig = px.bar(train_rate, x='DepartmentName', y='Trainings_Completed', 
             title="Completed Trainings by Department")
st.plotly_chart(fig, use_container_width=True)

# Awareness Score Distribution
awareness = df_training[df_training['TrainingType']=="Cybersecurity Awareness"]
fig2 = px.histogram(awareness, x="Score", nbins=20, title="Awareness Survey Score Distribution")
st.plotly_chart(fig2, use_container_width=True)

# ---- INCIDENTS & THREATS ----
st.subheader("Incident Analytics")
# Incidents by type
fig3 = px.bar(filtered_incidents['IncidentType'].value_counts().reset_index(), 
              x='index', y='IncidentType', 
              title="Incident Types Reported", labels={'index': 'Incident Type', 'IncidentType':'Count'})
st.plotly_chart(fig3, use_container_width=True)

# Incidents over time
filtered_incidents['DateReported'] = pd.to_datetime(filtered_incidents['DateReported'])
trend = filtered_incidents.groupby(filtered_incidents['DateReported'].dt.to_period("M")).size()
fig4 = px.line(x=trend.index.astype(str), y=trend.values, markers=True, 
               title="Incidents Reported Over Time", labels={'x': 'Month', 'y': 'Incidents'})
st.plotly_chart(fig4, use_container_width=True)

# Severity heatmap (Department x Severity)
dept_sev = pd.crosstab(filtered_incidents['DepartmentID'], filtered_incidents['Severity'])
dept_sev = dept_sev.rename(index=df_dept.set_index('DepartmentID')['DepartmentName'].to_dict())
fig5 = px.imshow(dept_sev, text_auto=True, aspect="auto", title="Incident Severity Heatmap (by Dept.)")
st.plotly_chart(fig5, use_container_width=True)

# ---- SECURITY EVENTS ----
st.subheader("Security Events & Agentic AI")
# Event type breakdown
fig6 = px.bar(df_events['EventType'].value_counts().reset_index(), 
              x='index', y='EventType', 
              title="Security Events by Type", labels={'index': 'Event Type', 'EventType': 'Count'})
st.plotly_chart(fig6, use_container_width=True)

# Threat Level pie
fig7 = px.pie(df_events, names='ThreatLevel', title="Event Threat Level Distribution")
st.plotly_chart(fig7, use_container_width=True)

# ---- AGENTIC AI LOGS ----
st.subheader("Agentic AI Action Log")
ai_action_counts = df_ai['ActionTaken'].value_counts().reset_index()
fig8 = px.bar(ai_action_counts, x='index', y='ActionTaken', title="Agentic AI Actions Taken", 
              labels={'index':'Action', 'ActionTaken':'Count'})
st.plotly_chart(fig8, use_container_width=True)

# Confidence by action
fig9 = px.box(df_ai, x='ActionTaken', y='DecisionConfidence', 
              title="AI Decision Confidence by Action")
st.plotly_chart(fig9, use_container_width=True)

# ---- BLOCKCHAIN AUDIT ----
st.subheader("Blockchain Audit Log")
st.dataframe(df_bc.sample(20), use_container_width=True)

# ---- POLICY & GOVERNANCE ----
st.subheader("Security Policy Registry (Latest 12)")
st.dataframe(df_policy.sort_values('DateActive', ascending=False).head(12), use_container_width=True)

# ---- EXPLORATORY: EMPLOYEE PROFILE VIEWER ----
st.subheader("Employee Profile Explorer")
emp_sel = st.selectbox("Select Employee:", filtered_emp['Name'])
emp_id = filtered_emp[filtered_emp['Name']==emp_sel]['EmployeeID'].values[0]
emp_info = filtered_emp[filtered_emp['EmployeeID']==emp_id].T
st.write(emp_info)
st.markdown("**Trainings:**")
st.dataframe(df_training[df_training['EmployeeID']==emp_id])
st.markdown("**Incidents:**")
st.dataframe(df_incidents[df_incidents['EmployeeID']==emp_id])

st.success("Dashboard loaded! Filter, explore, and analyze your synthetic SOC-as-a-Service data in real time.")
