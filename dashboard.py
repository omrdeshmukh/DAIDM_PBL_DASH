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
from data_cleaning import clean_incidents_data, clean_employee_ml_data

st.set_page_config(page_title="CyberSOC-aaS Executive Dashboard", layout="wide")
st.title("Blockchain-Enabled Agentic AI Cybersecurity Platform")

# DATA LOAD LOGIC (Upload or Backend)
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

df_emp = dfs['employees']
df_dept = dfs['departments']
df_roles = dfs['roles']
df_training = dfs['training']
df_incidents = dfs['incidents']
df_events = dfs['security_events']
df_ai = dfs['agentic_ai_log']
df_bc = dfs['blockchain_audit_log']
df_policy = dfs['policy']

# Data cleaning for ML and detailed analytics
df_inc = clean_incidents_data(df_incidents, df_emp, df_training)
df_emp_ml = clean_employee_ml_data(df_emp, df_incidents, df_training)
# ---- MAIN TABS: Each tab covers a critical aspect ----
tabs = st.tabs([
    "1️⃣ Executive Overview",
    "2️⃣ Incident Trends & Volume",
    "3️⃣ Training & Human Risk",
    "4️⃣ Departmental Risk & Heatmaps",
    "5️⃣ Incident Types & Attack Vectors",
    "6️⃣ Agentic AI & Response",
    "7️⃣ Policy & Compliance",
    "8️⃣ Employee/Asset Explorer",
    "9️⃣ ML Analytics & Risk Segmentation",
    "🔟 Root Cause & Deep Dive Explorer"
])

# ========== TAB 1: EXECUTIVE OVERVIEW (10 Dashboards) ==========
with tabs[0]:
    st.header("Executive Overview: Cyber Risk at a Glance")
    st.markdown("""
    _This tab is designed for the CEO, CTO, CXO, and Board: instantly see if the organization is getting safer, and which areas demand immediate attention. Every dashboard below is actionable and can be filtered for real-time decisions._
    """)

    # 1. Key Metrics
    st.subheader("1. Top-Level Risk Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Employees", df_emp.shape[0])
    col2.metric("Open Incidents", (df_incidents['Status'] != "Closed").sum())
    col3.metric("Critical Incidents", (df_incidents['Severity'] == "Critical").sum())
    col4.metric("Closed Incidents", (df_incidents['Status'] == "Closed").sum())
    col5.metric("Trainings Completed", (df_training['Status'] == "Completed").sum())
    st.caption("**Why this matters:** Sudden spikes in open or critical incidents require board-level intervention.")

    # 2. Incident Closure Rate
    st.subheader("2. Incident Closure Rate Over Time")
    closure = df_incidents.copy()
    closure['Month'] = pd.to_datetime(closure['DateReported']).dt.to_period('M')
    closure_rate = closure.groupby('Month')['Status'].apply(lambda x: (x == 'Closed').mean())
    closure_rate_df = pd.DataFrame({'Month': closure_rate.index.astype(str), 'ClosureRate': closure_rate.values})
    fig = px.line(closure_rate_df, x='Month', y='ClosureRate', markers=True, title="Incident Closure Rate (%)", labels={'Month': 'Month', 'ClosureRate': 'Closure Rate'})
    st.plotly_chart(fig, use_container_width=True, key="overview_incident_closure_rate")
    st.caption("**CEO insight:** A closure rate below 85% signals backlog; consider surge IT resources.")

    # 3. Training Completion Over Time
    st.subheader("3. Training Completion Trend")
    training = df_training.copy()
    training['Month'] = pd.to_datetime(training['DateCompleted'], errors='coerce').dt.to_period('M')
    tc_trend = training.groupby('Month')['Status'].apply(lambda x: (x=="Completed").mean())
    tc_trend_df = pd.DataFrame({'Month': tc_trend.index.astype(str), 'CompletionRate': tc_trend.values})
    fig2 = px.line(tc_trend_df, x='Month', y='CompletionRate', markers=True, title="Monthly Training Completion Rate", labels={'Month':'Month', 'CompletionRate':'Completion Rate'})
    st.plotly_chart(fig2, use_container_width=True, key="overview_training_completion")
    st.caption("**Board focus:** Monitor dips. Surge completions after incidents = 'learning after breach', not before.")

    # 4. Current Policy Compliance
    st.subheader("4. Current Policy Compliance")
    policy_latest = df_policy.sort_values('DateActive', ascending=False).head(1)
    st.dataframe(policy_latest, use_container_width=True)
    st.caption("**Board duty:** Ensure all policies are current and enforced. AI-enforced? Highlight for innovation story.")

    # 5. Risk By Region/Location
    st.subheader("5. Risk by Office Location")
    emp_loc_counts = df_emp['Location'].value_counts().reset_index()
    emp_loc_counts.columns = ['Location','EmployeeCount']
    inc_loc = df_incidents.merge(df_emp[['EmployeeID','Location']], on='EmployeeID', how='left')
    inc_per_loc = inc_loc.groupby('Location').size().reset_index(name='IncidentCount')
    merged = emp_loc_counts.merge(inc_per_loc, on='Location', how='outer').fillna(0)
    fig3 = px.scatter(merged, x='EmployeeCount', y='IncidentCount', text='Location',
                      title="Location Risk: Incidents vs Employees", labels={'x':'Employees','y':'Incidents'})
    st.plotly_chart(fig3, use_container_width=True, key="overview_location_risk")
    st.caption("**Executive view:** High incident-per-employee ratio? Remote sites often have higher risk.")

    # 6. Top 5 Unresolved Incidents
    st.subheader("6. Top 5 Longest-Open Incidents")
    open_inc = df_incidents[df_incidents['Status'] != "Closed"]
    if not open_inc.empty:
        st.dataframe(open_inc[['IncidentID','IncidentType','DateReported','Severity','Status']].sort_values('DateReported').head(5), use_container_width=True)
    else:
        st.success("No open incidents! Your ops team is responsive.")
    st.caption("**Why track:** Unresolved incidents are regulatory and reputational landmines.")

    # 7. Incidents by Employee Role
    st.subheader("7. Incident Volume by Role")
    role_map = df_roles.set_index('RoleID')['RoleName'].to_dict()
    emp_roles = df_emp[['EmployeeID','RoleID']].copy()
    emp_roles['Role'] = emp_roles['RoleID'].map(role_map)
    incidents_by_role = df_incidents.merge(emp_roles, on='EmployeeID', how='left')
    role_counts = incidents_by_role['Role'].value_counts().reset_index()
    role_counts.columns = ['Role','IncidentCount']
    fig4 = px.bar(role_counts, x='Role', y='IncidentCount', title="Incidents by Employee Role")
    st.plotly_chart(fig4, use_container_width=True, key="overview_incident_by_role")
    st.caption("**Executive lens:** Frontline staff typically have more incidents, but executive/C-suite incidents = existential risk.")

    # 8. Awareness Score Distribution (with filter)
    st.subheader("8. Awareness Score Distribution")
    awareness = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness']
    if awareness.empty:
        st.warning("No Cybersecurity Awareness data found.")
        min_score, max_score = 0, 100
    else:
        min_score, max_score = int(awareness['Score'].min()), int(awareness['Score'].max())

    score_range = st.slider("Awareness Score Range (Filter)", min_value=min_score, max_value=max_score,
                            value=(min_score, max_score), step=1)
    filtered_awareness = awareness[(awareness['Score'] >= score_range[0]) & (awareness['Score'] <= score_range[1])]
    fig5 = px.histogram(filtered_awareness, x="Score", nbins=20, color="Status", title="Filtered Awareness Scores")
    st.plotly_chart(fig5, use_container_width=True, key="overview_awareness_score_hist")
    st.caption("**Actionable:** Use filter to spot gaps and tailor interventions.")

    # 9. Incidents By AI vs Human Detection
    st.subheader("9. Detection Channel Breakdown")
    ai_vs_human = df_events['DetectedBy'].value_counts().reset_index()
    ai_vs_human.columns = ['DetectedBy','EventCount']
    fig6 = px.pie(ai_vs_human, names="DetectedBy", values="EventCount", title="Events: AI vs Human Detection")
    st.plotly_chart(fig6, use_container_width=True, key="overview_ai_vs_human")
    st.caption("**Board focus:** Growing % of AI detection = modern, scalable cyber program.")

    # 10. Incidents Over Time by Severity (multi-select)
    st.subheader("10. Monthly Incidents by Severity")
    sev_select = st.multiselect("Select Severity", options=df_incidents['Severity'].unique().tolist(), default=list(df_incidents['Severity'].unique()))
    df_sev_trend = df_incidents[df_incidents['Severity'].isin(sev_select)].copy()
    df_sev_trend['Month'] = pd.to_datetime(df_sev_trend['DateReported']).dt.to_period('M')
    monthly_sev = df_sev_trend.groupby(['Month','Severity']).size().reset_index(name='Count')
    monthly_sev['Month'] = monthly_sev['Month'].astype(str) 
    fig7 = px.line(monthly_sev, x='Month', y='Count', color='Severity', markers=True, title="Incidents Over Time (by Severity)")
    st.plotly_chart(fig7, use_container_width=True, key="overview_monthly_by_severity")
    st.caption("**Actionable:** Spikes in 'Critical' or 'High' must trigger crisis comms and action.")

# ========== TAB 2: INCIDENT TRENDS & VOLUME ==========
with tabs[1]:
    st.header("Incident Trends & Volume: Patterns and Anomalies")
    st.markdown("""
    _This tab helps executives, IT, and risk managers spot patterns in incident reporting and response, drilling down into time, type, source, and more._
    """)

    # 1. Incidents per Week/Month (toggle)
    st.subheader("1. Incidents Over Time")
    time_gran = st.radio("Select Time Granularity", ["Month", "Week"])
    if time_gran == "Month":
        trend = df_incidents.groupby(pd.to_datetime(df_incidents['DateReported']).dt.to_period('M')).size()
    else:
        trend = df_incidents.groupby(pd.to_datetime(df_incidents['DateReported']).dt.to_period('W')).size()
        trend_df = pd.DataFrame({'Time': trend.index.astype(str), 'Count': trend.values})
        fig = px.line(trend_df, x='Time', y='Count', markers=True, title=f"Incidents Reported per {time_gran}", labels={'Time': time_gran, 'Count': 'Number of Incidents'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"**Why it matters:** Trends reveal breach campaigns, detection gaps, or changes in staff behavior.")

    # 2. Incidents by Day of Week
    st.subheader("2. Incident Frequency by Day of Week")
    df_incidents['DayOfWeek'] = pd.to_datetime(df_incidents['DateReported']).dt.day_name()
    by_day = df_incidents['DayOfWeek'].value_counts().reindex([
        'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'
    ], fill_value=0)
    by_day_df = pd.DataFrame({'DayOfWeek': by_day.index, 'Count': by_day.values})
    fig2 = px.bar(by_day_df, x='DayOfWeek', y='Count', title="Incidents by Day of Week")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**Executive takeaway:** Clusters on certain days may signal process/shift/monitoring issues.")

    # 3. Incident Report Lag (Reporting Speed)
    st.subheader("3. Incident Report Lag (Speed to Response)")
    if 'DateResolved' in df_incidents.columns:
        df_incidents['DateReported'] = pd.to_datetime(df_incidents['DateReported'])
        df_incidents['DateResolved'] = pd.to_datetime(df_incidents['DateResolved'], errors='coerce')
        df_incidents['LagDays'] = (df_incidents['DateResolved'] - df_incidents['DateReported']).dt.days
        fig3 = px.box(df_incidents, y='LagDays', points="all", title="Time to Incident Resolution (Days)")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("**Board action:** Outliers with long lags = process or resource problems.")

    # 4. Open vs Closed Incidents by Department
    st.subheader("4. Open vs Closed Incidents (by Department)")
    dept_status = df_incidents.groupby(['DepartmentID', 'Status']).size().reset_index(name='Count')
    dept_status['Department'] = dept_status['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName'])
    fig4 = px.bar(dept_status, x='Department', y='Count', color='Status', barmode='group', title="Incident Status by Department")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("**For management:** Departments with high % open incidents may need support or process overhaul.")

    # 5. Incident Source: Internal vs External
    st.subheader("5. Incident Source: Internal vs External")
    # Assume incidents with employee ID in top mgmt/IT are internal, else external (customize for real org)
    staff_roles = df_emp.set_index('EmployeeID')['RoleID'].to_dict()
    df_incidents['SourceType'] = df_incidents['EmployeeID'].map(lambda eid: 'Internal' if staff_roles.get(eid, 0) in [1,2,3] else 'External')
    fig5 = px.pie(df_incidents, names='SourceType', title="Incident Source Breakdown")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("**Risk context:** External incidents = perimeter risk; internal = HR/process/privilege risk.")

    # 6. Incidents by Time of Day
    st.subheader("6. Incidents by Hour")
    df_incidents['Hour'] = pd.to_datetime(df_incidents['DateReported']).dt.hour
    fig6 = px.histogram(df_incidents, x='Hour', nbins=24, title="Incident Volume by Hour of Day")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**Board takeaway:** Peaks at night = need for 24x7 monitoring or process improvement.")

    # 7. Top 10 Employees by Incident Count
    st.subheader("7. Top 10 Employees by Incidents")
    emp_counts = df_incidents['EmployeeID'].value_counts().head(10)
    emp_names = df_emp.set_index('EmployeeID')['Name'].to_dict()
    emp_labels = [emp_names.get(eid, eid) for eid in emp_counts.index]
    emp_counts_df = pd.DataFrame({'Employee': emp_labels, 'Count': emp_counts.values})
    fig7 = px.bar(emp_counts_df, x='Employee', y='Count', title="Employees with Most Incidents")
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("**Management:** Use for HR engagement or to find training/monitoring needs.")

    # 8. Incidents by Status and Severity (stacked)
    st.subheader("8. Incidents by Status and Severity")
    stacked = df_incidents.groupby(['Status','Severity']).size().reset_index(name='Count')
    fig8 = px.bar(stacked, x='Status', y='Count', color='Severity', barmode='stack', title="Incident Status/Severity Breakdown")
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("**Exec summary:** See if open incidents are trending toward higher severity.")

    # 9. AI vs Human Detected (Over Time)
    st.subheader("9. AI vs Human Detection Over Time")
    df_events['EventDate'] = pd.to_datetime(df_events['Timestamp'], errors='coerce').dt.to_period('M')
    ai_trend = df_events.groupby(['EventDate','DetectedBy']).size().reset_index(name='Count')
    fig9 = px.line(ai_trend, x='EventDate', y='Count', color='DetectedBy', markers=True,
                   title="Detection Channel Over Time")
    st.plotly_chart(fig9, use_container_width=True)
    st.caption("**Innovation metric:** Board can track AI adoption and human review trends.")

    # 10. Custom Time Filter: Rolling Window
    st.subheader("10. Incidents in Rolling Time Window")
    days_window = st.slider("Select Rolling Window (Days)", min_value=7, max_value=90, value=30)
    latest = df_incidents[pd.to_datetime(df_incidents['DateReported']) >= (pd.to_datetime('today') - pd.Timedelta(days=days_window))]
    fig10 = px.histogram(latest, x='Severity', color='Status', title=f"Incidents in Last {days_window} Days (by Severity & Status)")
    st.plotly_chart(fig10, use_container_width=True)
    st.caption("**Why:** Zoom in for current quarter/month. Useful for monthly Board/IT reviews.")
# ========== TAB 3: TRAINING & HUMAN RISK ==========
with tabs[2]:
    st.header("Training, Human Error, and Risk Mitigation")
    st.markdown("""
    _This tab reveals how well the workforce is prepared—linking training to incident reduction and identifying weak links._
    """)

    # 1. Overall Training Completion Rate
    st.subheader("1. Training Completion Rate")
    total_trainings = len(df_training)
    completed = (df_training['Status'] == "Completed").sum()
    st.progress(completed / total_trainings if total_trainings else 1)
    st.write(f"Completed: {completed} / {total_trainings}")
    st.caption("**Business point:** Board wants to see >90% completion for regulatory compliance.")

    # 2. Training Completion by Department
    st.subheader("2. Completion by Department")
    df_training_emp = df_training.merge(df_emp[['EmployeeID','DepartmentID']], on='EmployeeID', how='left')
    comp_by_dept = df_training_emp.groupby('DepartmentID')['Status'].apply(lambda x: (x=="Completed").mean()).reset_index()
    comp_by_dept['Department'] = comp_by_dept['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName'])
    fig = px.bar(comp_by_dept, x='Department', y='Status', title="Training Completion Rate by Department", labels={'Status':'Completion Rate'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**For CISO:** Pinpoint which teams need more compliance pressure.")

    # 3. Phishing Simulation Scores
    st.subheader("3. Phishing Simulation Scores")
    phishing = df_training[df_training['TrainingType'] == 'Phishing Simulation']
    if not phishing.empty:
        fig2 = px.histogram(phishing, x="Score", nbins=10, color="Status", title="Phishing Simulation Score Distribution")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("**Board metric:** A shift left (low scores) = increased phishing risk; repeat test after intervention.")

    # 4. Avg. Awareness Score by Role
    st.subheader("4. Awareness by Role")
    avg_aw = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness'].merge(df_emp[['EmployeeID','RoleID']], on='EmployeeID')
    avg_aw = avg_aw.groupby('RoleID')['Score'].mean().reset_index()
    avg_aw['Role'] = avg_aw['RoleID'].map(df_roles.set_index('RoleID')['RoleName'])
    fig3 = px.bar(avg_aw, x='Role', y='Score', title="Avg Awareness Score by Role")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("**For Board:** Executives/managers must have high scores (target >85).")

    # 5. Number of Trainings per Employee (Gap analysis)
    st.subheader("5. Training Engagement Gap")
    train_counts = df_training.groupby('EmployeeID').size().reset_index(name='TrainingCount')
    fig4 = px.histogram(train_counts, x="TrainingCount", nbins=10, title="Number of Trainings per Employee")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("**Action:** Employees with <2 trainings per year = risk flag for HR review.")

    # 6. Failed Trainings
    st.subheader("6. Failed Training Analysis")
    failed = df_training[df_training['Status'] == 'Failed']
    failed_dept = failed.merge(df_emp[['EmployeeID','DepartmentID']], on='EmployeeID')
    failed_dept = failed_dept['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName']).value_counts()
    failed_dept_df = pd.DataFrame({'Department': failed_dept.index, 'Count': failed_dept.values})
    fig5 = px.bar(failed_dept_df, x='Department', y='Count', title="Failed Trainings by Department")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("**Board impact:** Persistent failures signal cultural/process gaps.")

    # 7. Correlation: Training Score vs. Incidents Caused
    st.subheader("7. Are Well-Trained Employees Safer?")
    awareness_score = df_training[df_training['TrainingType']=='Cybersecurity Awareness'].groupby('EmployeeID')['Score'].mean().reset_index()
    incidents_caused = df_incidents.groupby('EmployeeID').size().reset_index(name='IncidentsCaused')
    merged = awareness_score.merge(incidents_caused, on='EmployeeID', how='left').fillna({'IncidentsCaused':0})
    fig6 = px.scatter(merged, x='Score', y='IncidentsCaused', trendline='ols', title="Awareness Score vs Incidents Caused")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**Data-driven:** Higher scores should mean fewer incidents. If not, training is not working.")

    # 8. High-Risk Employees (low score & high incidents)
    st.subheader("8. High-Risk Employee List")
    high_risk = merged[(merged['Score'] < 60) & (merged['IncidentsCaused'] > 1)]
    risky_emp_names = df_emp.set_index('EmployeeID').loc[high_risk['EmployeeID']]['Name'].values if not high_risk.empty else []
    st.write("Employees with low awareness & high incidents:", risky_emp_names)
    st.caption("**Board focus:** These are your most urgent training/intervention targets.")

    # 9. Training Trend by Type (multi-select)
    st.subheader("9. Training Type Trend")
    train_types = df_training['TrainingType'].unique().tolist()
    type_select = st.multiselect("Filter Training Types", options=train_types, default=train_types)
    df_type_trend = df_training[df_training['TrainingType'].isin(type_select)].copy()
    df_type_trend['Month'] = pd.to_datetime(df_type_trend['DateCompleted'], errors='coerce').dt.to_period('M')
    type_trend = df_type_trend.groupby(['Month','TrainingType']).size().reset_index(name='Count')
    fig7 = px.line(type_trend, x='Month', y='Count', color='TrainingType', title="Trainings Completed by Type (Monthly)")
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("**Management:** Choose focus areas for the next training wave.")

    # 10. Certification Earned (impact)
    st.subheader("10. Cybersecurity Certifications Earned")
    if 'CertificateURL' in df_training.columns:
        cert_earned = df_training[df_training['CertificateURL'].notnull() & (df_training['CertificateURL'] != "")]
        cert_counts = cert_earned.groupby('TrainingType').size().reset_index(name='CertCount')
        fig8 = px.bar(cert_counts, x='TrainingType', y='CertCount', title="Certifications Earned by Type")
        st.plotly_chart(fig8, use_container_width=True)
        st.caption("**Board-level metric:** Certifications build external trust and readiness (insurance, audit).")
# ========== TAB 4: DEPARTMENTAL RISK & HEATMAPS ==========
with tabs[3]:
    st.header("Departmental Risk: Hotspots, Severity, and Patterns")
    st.markdown("""
    _This tab lets business leaders pinpoint which parts of the organization are most at risk, which ones are improving, and where to focus attention next quarter._
    """)

    # 1. Departmental Incident Volume
    st.subheader("1. Incident Volume by Department")
    dept_counts = df_incidents['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName']).value_counts()
    dept_counts_df = pd.DataFrame({'Department': dept_counts.index, 'Count': dept_counts.values})
    fig = px.bar(dept_counts_df, x='Department', y='Count', title="Incidents by Department")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Management:** Persistent top-3 risk departments should trigger targeted audits.")

    # 2. Departmental Incident Severity
    st.subheader("2. Severity Mix per Department")
    dept_sev = pd.crosstab(df_incidents['DepartmentID'], df_incidents['Severity'])
    dept_sev.index = dept_sev.index.map(df_dept.set_index('DepartmentID')['DepartmentName'].to_dict())
    fig2 = px.imshow(dept_sev, text_auto=True, aspect="auto", title="Incident Severity Heatmap (by Dept.)")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**Board insight:** Red cells = chronic risk. Drive root-cause analysis.")

    # 3. Severity Distribution (choose departments)
    st.subheader("3. Filtered Severity by Department")
    depts = dept_sev.index.tolist()
    dept_select = st.multiselect("Departments to Compare", depts, default=depts[:3])
    filtered = df_incidents[df_incidents['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName']).isin(dept_select)]
    fig3 = px.histogram(filtered, x='Severity', color='IncidentType', barmode='group', title="Severity by Incident Type (Filtered)")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("**Exec use:** Compare target departments to see if severity mix is improving.")

    # 4. Incidents Over Time by Department
    st.subheader("4. Monthly Incident Trend per Department")
    df_incidents['Month'] = pd.to_datetime(df_incidents['DateReported']).dt.to_period('M')
    dept_month = df_incidents.groupby(['Month','DepartmentID']).size().reset_index(name='Count')
    dept_month['Department'] = dept_month['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName'])
    fig4 = px.line(dept_month, x='Month', y='Count', color='Department', title="Monthly Incident Trend by Department")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("**For resource planning:** See where investment is paying off.")

    # 5. Top Employees Causing Incidents in Each Department
    st.subheader("5. Top Employees per Department")
    sel_dept = st.selectbox("Select Department for Drilldown", dept_counts.index)
    sel_dept_id = df_dept[df_dept['DepartmentName'] == sel_dept]['DepartmentID'].iloc[0]
    emp_ids = df_emp[df_emp['DepartmentID'] == sel_dept_id]['EmployeeID']
    top_emp = df_incidents[df_incidents['EmployeeID'].isin(emp_ids)]['EmployeeID'].value_counts().head(5)
    emp_names = df_emp.set_index('EmployeeID')['Name'].to_dict()
    fig5 = px.bar(x=[emp_names.get(e, e) for e in top_emp.index], y=top_emp.values, title=f"Top 5 Employees in {sel_dept}")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("**HR flag:** Chronic offenders may need individual intervention.")

    # 6. Departmental AI Detection Rate
    st.subheader("6. AI vs Human Detection (by Department)")
    events_dept = df_events.merge(df_emp[['EmployeeID','DepartmentID']], on='EmployeeID', how='left')
    ai_counts = events_dept.groupby(['DepartmentID','DetectedBy']).size().reset_index(name='Count')
    ai_counts['Department'] = ai_counts['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName'])
    fig6 = px.bar(ai_counts, x='Department', y='Count', color='DetectedBy', barmode='group', title="AI/Human Detection by Dept")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**Innovation lens:** Where is AI catching more, where do you still rely on humans?")

    # 7. Time to Resolve by Department
    st.subheader("7. Time to Resolve Incidents by Department")
    if 'DateResolved' in df_incidents.columns:
        df_incidents['DateResolved'] = pd.to_datetime(df_incidents['DateResolved'], errors='coerce')
        df_incidents['LagDays'] = (df_incidents['DateResolved'] - pd.to_datetime(df_incidents['DateReported'])).dt.days
        dept_lag = df_incidents.groupby('DepartmentID')['LagDays'].mean().reset_index()
        dept_lag['Department'] = dept_lag['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName'])
        fig7 = px.bar(dept_lag, x='Department', y='LagDays', title="Avg Days to Resolve (by Dept)")
        st.plotly_chart(fig7, use_container_width=True)
        st.caption("**Ops:** Longest lags = departments to target with workflow/process improvements.")

    # 8. Critical Incidents Map (if you have geo/location)
    st.subheader("8. Critical Incidents by Location")
    crit_locs = df_incidents[df_incidents['Severity'] == 'Critical'].merge(df_emp[['EmployeeID','Location']], on='EmployeeID', how='left')
    if not crit_locs.empty:
        fig8 = px.scatter(crit_locs, x='Location', y='IncidentType', color='Status', title="Critical Incidents by Location")
        st.plotly_chart(fig8, use_container_width=True)
        st.caption("**Board flag:** Repeat criticals in a location = systemic control breakdown.")

    # 9. Department Size vs Incident Rate
    st.subheader("9. Department Size vs Incidents")
    dept_size = df_emp['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName']).value_counts()
    dept_inc = dept_counts.reindex(dept_size.index, fill_value=0)
    fig9 = px.scatter(x=dept_size, y=dept_inc, text=dept_size.index,
                      labels={'x':'Department Size','y':'Incident Volume'},
                      title="Incident Volume vs Dept Size")
    st.plotly_chart(fig9, use_container_width=True)
    st.caption("**For Board:** High risk in small teams = more than just exposure, look for process/culture causes.")

    # 10. Departmental Risk Summary Table
    st.subheader("10. Departmental Risk Table")
    summary = pd.DataFrame({
        'Incidents': dept_counts,
        'CriticalInc': df_incidents[df_incidents['Severity']=='Critical']['DepartmentID']
            .map(df_dept.set_index('DepartmentID')['DepartmentName']).value_counts(),
        'AvgResolution': dept_lag.set_index('Department')['LagDays'] if 'dept_lag' in locals() else 0
    }).fillna(0).astype(int)
    st.dataframe(summary)
    st.caption("**For Board packs:** Summarize risk position for each major business unit.")
# ========== TAB 5: INCIDENT TYPES & ATTACK VECTORS ==========
with tabs[4]:
    st.header("Incident Types & Attack Vectors: Patterns & Response")
    st.markdown("""
    _This tab is for the CTO, CISO, and incident response leads—break down which threats hit hardest, where they're happening, and how they're being handled._
    """)

    # 1. Incidents by Type (with filters)
    st.subheader("1. Incident Volume by Type")
    type_counts = df_incidents['IncidentType'].value_counts()
    selected_types = st.multiselect("Filter Incident Types", options=type_counts.index, default=list(type_counts.index)[:3])
    filtered_inc = df_incidents[df_incidents['IncidentType'].isin(selected_types)]
    vc = filtered_inc['IncidentType'].value_counts().reset_index()
    vc.columns = ['IncidentType', 'Count']
    fig = px.bar(vc, x='IncidentType', y='Count', title="Filtered Incidents by Type")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Use:** Focus on top attack vectors for budget/board reporting.")

    # 2. Severity Mix per Type
    st.subheader("2. Severity Distribution per Incident Type")
    type_sev = pd.crosstab(df_incidents['IncidentType'], df_incidents['Severity'])
    type_select2 = st.multiselect("Choose Types", options=type_sev.index.tolist(), default=type_sev.index.tolist())
    fig2 = px.imshow(type_sev.loc[type_select2], text_auto=True, aspect="auto", title="Severity by Type (Selected)")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**For cyber ops:** Types with many criticals = priority for new controls.")

    # 3. Type Trend Over Time
    st.subheader("3. Incident Type Trends")
    df_incidents['Month'] = pd.to_datetime(df_incidents['DateReported']).dt.to_period('M')
    trend_type = df_incidents[df_incidents['IncidentType'].isin(selected_types)]
    trend = trend_type.groupby(['Month','IncidentType']).size().reset_index(name='Count')
    fig3 = px.line(trend, x='Month', y='Count', color='IncidentType', title="Incident Type Trend (Monthly)")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("**For strategic planning:** Is phishing rising? Malware falling? Inform next quarter's plan.")

    # 4. Incident Type by Detection Channel
    st.subheader("4. Detection Channel per Type")
    events_types = df_events.merge(df_incidents[['IncidentID','IncidentType']], left_on='EventID', right_on='IncidentID', how='left')
    type_detect = events_types.groupby(['IncidentType','DetectedBy']).size().reset_index(name='Count')
    fig4 = px.bar(type_detect, x='IncidentType', y='Count', color='DetectedBy', barmode='group', title="Detection by Type")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("**Board use:** Types only caught by humans = automation opportunity.")

    # 5. Phishing, Malware, Data Leak: Compare Side by Side
    st.subheader("5. High-Risk Vector Comparison")
    vecs = ['Phishing','Malware','DataLeak']
    for v in vecs:
        v_df = df_incidents[df_incidents['IncidentType']==v]
        st.write(f"**{v} incidents:** {len(v_df)}; Critical: {(v_df['Severity']=='Critical').sum()}; Median AI Response: {v_df['AIResponseTime'].median()}")
    st.caption("**Board use:** Use these stats for regulatory/insurance filings.")

    # 6. Resolution Time per Type
    st.subheader("6. Median Resolution Time by Type")
    if 'DateResolved' in df_incidents.columns:
        df_incidents['DateResolved'] = pd.to_datetime(df_incidents['DateResolved'], errors='coerce')
        df_incidents['LagDays'] = (df_incidents['DateResolved'] - pd.to_datetime(df_incidents['DateReported'])).dt.days
        res_time = df_incidents.groupby('IncidentType')['LagDays'].median().reset_index()
        fig6 = px.bar(res_time, x='IncidentType', y='LagDays', title="Median Resolution Time per Type")
        st.plotly_chart(fig6, use_container_width=True)
        st.caption("**CISO focus:** Types with longest lags need new playbooks/automation.")

    # 7. Top 10 Most Severe Incidents by Type
    st.subheader("7. Top 10 Severe Incidents (by Type)")
    severe_inc = df_incidents[df_incidents['Severity'].isin(['Critical','High'])]
    st.dataframe(severe_inc[['IncidentType','IncidentID','EmployeeID','DateReported','Status']].sort_values('DateReported', ascending=False).head(10), use_container_width=True)
    st.caption("**For Board packs:** Real-life stories are more compelling than stats.")

    # 8. Type by Department
    st.subheader("8. Incidents by Type & Department")
    type_dept = pd.crosstab(df_incidents['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName']),
                            df_incidents['IncidentType'])
    type_dept_select = st.multiselect("Departments", options=type_dept.index, default=type_dept.index)
    fig8 = px.imshow(type_dept.loc[type_dept_select], text_auto=True, aspect="auto", title="Incident Type Heatmap (by Dept)")
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("**Exec use:** Exposed departments for each vector = process or culture gap.")

    # 9. Incident Type Overlap with Policy Enforcement
    st.subheader("9. Policy Enforcement vs Incident Type")
    pol_type = df_policy.groupby('Name').size().reset_index(name='PolicyCount')
    st.dataframe(pol_type, use_container_width=True)
    st.caption("**Board:** Which attack vectors have policies? Gaps = exposure.")

    # 10. 3D Plot: Incidents by Type, Severity, AI Response
    st.subheader("10. 3D Incident Landscape")
    from plotly.graph_objs import Scatter3d, Layout, Figure
    type_map = {k: i for i, k in enumerate(df_incidents['IncidentType'].unique())}
    sev_map = {'Low':0, 'Medium':1, 'High':2, 'Critical':3}
    df_3d = df_incidents.copy()
    df_3d['TypeIdx'] = df_3d['IncidentType'].map(type_map)
    df_3d['SevIdx'] = df_3d['Severity'].map(sev_map)
    fig10 = px.scatter_3d(df_3d, x='TypeIdx', y='AIResponseTime', z='SevIdx',
                          color='IncidentType', hover_data=['IncidentID','DateReported','Severity'],
                          title="3D: Incident Type vs AI Response vs Severity")
    fig10.update_layout(scene = dict(
        xaxis_title='Incident Type',
        yaxis_title='AI Response (min)',
        zaxis_title='Severity'))
    st.plotly_chart(fig10, use_container_width=True)
    st.caption("**CISO/Board:** Spot patterns—are some vectors consistently slow to detect, or always more severe?")
# ========== TAB 6: AGENTIC AI & RESPONSE ==========
with tabs[5]:
    st.header("Agentic AI Automation & Incident Response")
    st.markdown("""
    _Track how the organization's AI engine is responding, escalating, and learning. This tab is for the CTO, SOC leads, and audit/compliance teams to demonstrate AI-driven improvement and monitor for errors or escalation needs._
    """)

    # 1. AI Actions Over Time
    st.subheader("1. AI Actions: Monthly Trend")
    df_ai['Month'] = pd.to_datetime(df_ai['Timestamp'], errors='coerce').dt.to_period('M')
    ai_monthly = df_ai.groupby('Month').size().reset_index(name='ActionCount')
    fig = px.line(ai_monthly, x='Month', y='ActionCount', markers=True, title="Monthly AI Actions")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Exec insight:** Is AI engagement rising? Flatlining may mean plateaued adoption.")

    # 2. AI Decision Confidence Distribution
    st.subheader("2. AI Decision Confidence")
    fig2 = px.histogram(df_ai, x='DecisionConfidence', nbins=20, color='ActionTaken', title="AI Confidence by Action")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**CISO:** Low confidence clusters = consider more human review or retrain model.")

    # 3. Human Escalation Rate
    st.subheader("3. Human Escalation Rate Over Time")
    df_ai['EscalatedToHuman'] = df_ai['EscalatedToHuman'].astype(str)
    esc_rate = df_ai.groupby(['Month','EscalatedToHuman']).size().reset_index(name='Count')
    fig3 = px.bar(esc_rate, x='Month', y='Count', color='EscalatedToHuman', barmode='group', title="Escalations to Human Analyst")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("**Board:** Steadily decreasing escalations signals AI maturity and trust.")

    # 4. AI Outcomes by Action
    st.subheader("4. AI Outcomes by Action")
    outcome_counts = df_ai.groupby(['ActionTaken','Outcome']).size().reset_index(name='Count')
    fig4 = px.bar(outcome_counts, x='ActionTaken', y='Count', color='Outcome', barmode='group', title="AI Outcomes by Action")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("**Management:** Frequent 'Failed' outcomes = retrain or redesign automation.")

    # 5. Average Response Time by AI Action
    st.subheader("5. Avg Response Time by Action")
    ai_rt = df_ai.groupby('ActionTaken')['DecisionConfidence'].mean().reset_index(name='AvgConfidence')
    fig5 = px.bar(ai_rt, x='ActionTaken', y='AvgConfidence', title="Avg Confidence by Action")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("**Audit:** Are some actions consistently less confident? Focus human review there.")

    # 6. Escalation Outcomes (Success/Fail)
    st.subheader("6. Escalation Success Rate")
    esc_outcomes = df_ai[df_ai['EscalatedToHuman']=='True'].groupby('Outcome').size().reset_index(name='Count')
    fig6 = px.pie(esc_outcomes, names='Outcome', values='Count', title="Human Escalation Outcomes")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**Audit/QA:** Are humans correcting AI errors or confirming it?")

    # 7. Most Common AI Actions
    st.subheader("7. Most Frequent AI Actions")
    action_counts = df_ai['ActionTaken'].value_counts().reset_index()
    action_counts.columns = ['ActionTaken','Count']
    fig7 = px.bar(action_counts, x='ActionTaken', y='Count', title="Top AI Actions")
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("**Exec:** What is AI actually doing most? Does this align with top risk")

# ========== TAB 7: POLICY & COMPLIANCE ==========
with tabs[6]:
    st.header("Policy, Governance & Compliance")
    st.markdown("""
    _A board/exec view of policy coverage, change, and effectiveness, and a regulatory audit toolkit. Demonstrate and drive continuous improvement._
    """)

    # 1. Active Policy List
    st.subheader("1. Active Policy Registry")
    st.dataframe(df_policy.sort_values('DateActive', ascending=False), use_container_width=True)
    st.caption("**Audit:** Are all policies current, version-controlled, and accessible?")

    # 2. Policy Change Trend
    st.subheader("2. Policy Change Frequency")
    df_policy['DateActive'] = pd.to_datetime(df_policy['DateActive'], errors='coerce')
    policy_month = df_policy.groupby(df_policy['DateActive'].dt.to_period('M')).size()
    fig2 = px.line(x=policy_month.index.astype(str), y=policy_month.values, markers=True,
                   title="Policy Changes (Monthly)", labels={'x':'Month','y':'Policy Updates'})
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**Board:** Surges in updates = response to threats or audit findings.")

    # 3. Policy Enforcement by AI/Human/Hybrid
    st.subheader("3. Enforcement Ownership")
    fig3 = px.pie(df_policy, names='EnforcementType', title="Policy Enforcement Channel")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("**For innovation:** % AI-enforced shows forward-leaning, scalable program.")

    # 4. Policy by Coverage Domain
    st.subheader("4. Policy by Domain")
    if 'Domain' in df_policy.columns:
        fig4 = px.bar(df_policy['Domain'].value_counts().reset_index(),
                      x='index', y='Domain', title="Policies by Domain", labels={'index':'Domain','Domain':'Policy Count'})
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("**Management:** Over- or under-weighted areas = exposure or over-regulation.")

    # 5. Policy-Linked Incident Types
    st.subheader("5. Policy Linked to Incident Types")
    if 'IncidentType' in df_policy.columns:
        pol_inc = df_policy['IncidentType'].value_counts()
        pol_inc_df = pol_inc.reset_index()
        pol_inc_df.columns = ['IncidentType', 'Count']
        fig5 = px.bar(pol_inc_df, x='IncidentType', y='Count', title="Policies Linked to Incident Types")
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("**Audit:** Unlinked types = exposure. Update policies to match current threats.")

    # 6. Policy Effectiveness: Incidents Pre/Post Update
    st.subheader("6. Policy Effectiveness (Pre/Post Update)")
    # Example: Compare incident rate before/after most recent policy update
    most_recent = df_policy['DateActive'].max()
    before = df_incidents[pd.to_datetime(df_incidents['DateReported']) < most_recent]
    after = df_incidents[pd.to_datetime(df_incidents['DateReported']) >= most_recent]
    fig6 = px.histogram(pd.DataFrame({'Period':['Before']*len(before)+['After']*len(after),
                                      'Incident': [1]*len(before) + [1]*len(after)}),
                        x='Period', color='Period', title="Incidents Before/After Latest Policy")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**C-suite:** Use as ROI argument for policy change.")

    # 7. Policy Review Schedule Adherence
    st.subheader("7. Policy Review Cadence")
    if 'ReviewFrequency' in df_policy.columns:
        freq_counts = df_policy['ReviewFrequency'].value_counts()
        fig7 = px.bar(freq_counts, title="Review Frequency of Policies")
        st.plotly_chart(fig7, use_container_width=True)
        st.caption("**Board:** Stale policies = audit/regulatory risk.")

    # 8. Policy Ownership: By Person or Function
    st.subheader("8. Policy Owner Distribution")
    if 'Owner' in df_policy.columns:
        owner_counts = df_policy['Owner'].value_counts().reset_index()
        fig8 = px.bar(owner_counts, x='index', y='Owner', title="Policy Owner Breakdown")
        st.plotly_chart(fig8, use_container_width=True)
        st.caption("**For accountability:** Who's responsible for gaps or outdated policies?")

    # 9. Policies Covering High-Risk Departments
    st.subheader("9. High-Risk Department Policy Coverage")
    risk_depts = dept_counts.head(3).index.tolist()
    if 'Department' in df_policy.columns:
        covered = df_policy[df_policy['Department'].isin(risk_depts)]
        st.dataframe(covered, use_container_width=True)
        st.caption("**Board:** Top-risk areas should have most/strongest policy coverage.")

    # 10. Policy-Triggered Blockchain Events
    st.subheader("10. Blockchain Events Tied to Policy Updates")
    if 'PolicyID' in df_bc.columns:
        policy_events = df_bc['PolicyID'].value_counts()
        fig10 = px.bar(policy_events, title="Blockchain Events per Policy")
        st.plotly_chart(fig10, use_container_width=True)
        st.caption("**Audit/Compliance:** Immutable record of all policy updates for audit trail.")
# ========== TAB 8: EMPLOYEE/ASSET EXPLORER ==========
with tabs[7]:
    st.header("Employee & Asset Risk Explorer")
    st.markdown("""
    _Drill into any employee, device, or asset to view cyber risk, performance, incidents, and training. A favorite for HR, CISO, and IT ops!_
    """)

    # 1. Select Employee/Asset
    st.subheader("1. Select Employee/Asset for Profile")
    emp_names = df_emp['Name'].tolist()
    selected_emp = st.selectbox("Select Employee", emp_names)
    emp_row = df_emp[df_emp['Name'] == selected_emp].iloc[0]
    st.write("### Employee Profile")
    st.json(emp_row.to_dict())

    # 2. Employee Training Record
    st.subheader("2. Trainings Completed by Employee")
    emp_train = df_training[df_training['EmployeeID'] == emp_row['EmployeeID']]
    st.dataframe(emp_train, use_container_width=True)
    st.caption("**HR use:** Track recency, gaps, and failed attempts.")

    # 3. Employee Incident History
    st.subheader("3. Incident History")
    emp_inc = df_incidents[df_incidents['EmployeeID'] == emp_row['EmployeeID']]
    st.dataframe(emp_inc, use_container_width=True)
    st.caption("**CISO/HR:** Repeated incidents = training or behavioral issue.")

    # 4. Recent Security Events
    st.subheader("4. Security Event Log")
    emp_ev = df_events[df_events['EmployeeID'] == emp_row['EmployeeID']]
    st.dataframe(emp_ev.head(10), use_container_width=True)
    st.caption("**IT:** See recent alert/monitoring events for this asset.")

    # 5. Employee Risk Score (custom formula)
    st.subheader("5. Employee Risk Score")
    # Example risk formula: low awareness + high incidents = high risk
    score = emp_train[emp_train['TrainingType']=='Cybersecurity Awareness']['Score'].mean() if not emp_train.empty else 0
    num_inc = len(emp_inc)
    risk_score = (100 - score) + (num_inc * 20)
    st.metric("Estimated Risk Score", f"{risk_score:.1f}")
    st.caption("**HR/IT:** High scores = higher monitoring, potential privilege reduction.")

    # 6. Risk Score Distribution (all employees)
    st.subheader("6. Employee Risk Score Distribution")
    all_emp = df_emp.copy()
    all_emp['Score'] = all_emp['EmployeeID'].map(
        df_training[df_training['TrainingType']=='Cybersecurity Awareness'].groupby('EmployeeID')['Score'].mean().to_dict()
    ).fillna(0)
    all_emp['Incidents'] = all_emp['EmployeeID'].map(df_incidents['EmployeeID'].value_counts()).fillna(0)
    all_emp['RiskScore'] = (100 - all_emp['Score']) + (all_emp['Incidents'] * 20)
    fig6 = px.histogram(all_emp, x='RiskScore', nbins=15, title="Distribution of Employee Risk Scores")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**Exec:** The right tail is your biggest insider risk cohort.")

    # 7. Asset Explorer: Select Device/Asset (if applicable)
    if 'AssetID' in df_events.columns:
        st.subheader("7. Asset Risk Drilldown")
        assets = df_events['AssetID'].unique().tolist()
        selected_asset = st.selectbox("Select Asset", assets)
        asset_ev = df_events[df_events['AssetID'] == selected_asset]
        st.dataframe(asset_ev.head(10), use_container_width=True)
        st.caption("**IT/OT:** Spot abnormal asset event patterns.")

    # 8. Time Since Last Training
    st.subheader("8. Days Since Last Training")
    last_train = emp_train['DateCompleted'].max() if not emp_train.empty else "Never"
    days_since = (pd.to_datetime('today') - pd.to_datetime(last_train)).days if last_train != "Never" else None
    st.write(f"Days since last training: {days_since if days_since is not None else 'Never trained'}")
    st.caption("**HR:** Longest gaps = priority for training assignment.")

    # 9. Most Common Incident Types for Employee
    st.subheader("9. Incident Types for This Employee")
    if not emp_inc.empty:
        inc_types = emp_inc['IncidentType'].value_counts().reset_index()
        inc_types.columns = ['IncidentType','Count']
        fig9 = px.bar(inc_types, x='IncidentType', y='Count', title="Employee's Incidents by Type")
        st.plotly_chart(fig9, use_container_width=True)
    st.caption("**Behavioral:** Repeats in same type = not learning or not reporting.")

    # 10. Incident Timeline for Employee
    st.subheader("10. Incident Timeline")
    if not emp_inc.empty:
        emp_inc['DateReported'] = pd.to_datetime(emp_inc['DateReported'])
        fig10 = px.line(emp_inc, x='DateReported', y='Severity', title="Incident Timeline")
        st.plotly_chart(fig10, use_container_width=True)
    st.caption("**Exec:** Bursts in timeline = stress, role change, or process issue.")
# ========== TAB 9: ML ANALYTICS & RISK SEGMENTATION ==========
with tabs[8]:
    st.header("ML Analytics & Risk Segmentation")
    st.markdown("""
    _This tab brings advanced data science and AI to the C-suite: see clusters, drivers of risk, prediction, and actionable feature importance. All results are interactive and explained in plain English for execs._
    """)

    # 1. 3D K-Means Clustering (Risk Groups)
    st.subheader("1. 3D K-Means Employee Risk Clustering")
    cluster_features = st.multiselect("Clustering Features", ['Score', 'IncidentsCaused', 'TrainingCount'], default=['Score','IncidentsCaused','TrainingCount'])
    max_k = st.slider("Elbow/Silhouette: Max clusters (K)", min_value=3, max_value=10, value=5)
    if len(cluster_features) >= 3:
        X = df_emp_ml[cluster_features].dropna().to_numpy()
        inertia = get_elbow_curve(X, max_k=max_k)
        sil_scores = get_silhouette_scores(X, max_k=max_k)
        st.plotly_chart(plot_elbow_curve(inertia, max_k=max_k), use_container_width=True)
        st.plotly_chart(plot_silhouette_curve(sil_scores, max_k=max_k), use_container_width=True)
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=max_k, value=3)
        df_clustered, kmeans, sil_score = kmeans_cluster(df_emp_ml, cluster_features, n_clusters)
        for col in ['EmployeeID', 'Name', 'DepartmentID', 'RoleID']:
            if col not in df_clustered.columns:
                df_clustered[col] = None
        fig1 = cluster_3d_plot(df_clustered, cluster_features[0], cluster_features[1], cluster_features[2])
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(f"**Interpretation:** Each 3D cluster is a distinct risk group. Right-click a point to see employee details.")

    # 2. Cluster Risk Table
    st.subheader("2. Cluster Summary Table")
    if 'cluster' in df_clustered.columns:
        clust_summary = df_clustered.groupby('cluster')[cluster_features].mean().reset_index()
        st.dataframe(clust_summary, use_container_width=True)
        st.caption("**Exec:** Focus on clusters with highest incidents/lowest awareness.")

    # 3. Classification Model Comparison (Accuracy)
    st.subheader("3. Incident Severity Prediction Accuracy")
    X_cols = ['AIResponseTime', 'DepartmentIDIdx', 'RoleIDIdx', 'LocationIdx', 'IncidentTypeIdx', 'Score']
    df_class = df_inc.dropna(subset=X_cols + ['Severity'])
    label_encoder = LabelEncoder()
    if df_class.empty:
        st.warning("Not enough data for classification.")
    else:
        models, y_test, label_encoder = train_classifiers(df_class, X_cols, 'Severity', label_encoder=label_encoder)
        accs = {name:score for name, (model, score, *_) in models.items()}
        fig2 = px.bar(x=list(accs.keys()), y=list(accs.values()), title="Model Accuracy Comparison")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("**For Board:** Which AI model predicts severe events best?")

    # 4. Confusion Matrix (toggle model)
    st.subheader("4. Confusion Matrix")
    model_choice = st.selectbox("Select Model", list(models.keys()))
    if model_choice in models:
        _, _, cm, *extras = models[model_choice]
        st.plotly_chart(plot_confusion(cm, labels=label_encoder.classes_, title=f"{model_choice} Confusion Matrix"), use_container_width=True)
        st.caption("**Exec:** Where do models most often mis-classify severity?")

    # 5. Feature Importance (Random Forest)
    st.subheader("5. Random Forest Feature Importances")
    if "RF" in models and models["RF"][3] is not None:
        st.bar_chart(pd.Series(models["RF"][3], index=X_cols))
        st.caption("**For Board:** The top feature is the best 'lever' for reducing incident severity.")

    # 6. Regression: Predicting Incident Volume
    st.subheader("6. Regression: Who Causes Most Incidents?")
    regression_features = st.multiselect("Regression Features", ['Score', 'RoleIdx', 'TrainingCount'], default=['Score','RoleIdx','TrainingCount'])
    target_reg = 'IncidentsCaused'
    df_reg = df_emp_ml
    if df_reg[regression_features + [target_reg]].dropna().empty:
        st.warning("Not enough clean data for regression.")
    else:
        regressors = train_regressors(df_reg, regression_features, target_reg)
        for name, (model, r2, coef) in regressors.items():
            st.write(f"**{name} Regression**: R2 = {r2:.2f}")
            if coef is not None:
                st.bar_chart(pd.Series(coef, index=regression_features))
        st.caption("**Exec:** Features with largest coefficients = main risk drivers.")

    # 7. Regression Residuals Plot (Random Forest)
    st.subheader("7. Random Forest Regression Residuals")
    if "RF" in regressors:
        y_true = df_reg[target_reg].values
        y_pred = regressors["RF"][0].predict(df_reg[regression_features])
        residuals = y_true - y_pred
        fig7 = px.histogram(residuals, nbins=30, title="Random Forest Regression Residuals")
        st.plotly_chart(fig7, use_container_width=True)
        st.caption("**Data Science:** Non-random pattern = missing driver or model improvement needed.")

    # 8. 3D Regression Plot (Risk Factors)
    st.subheader("8. 3D Regression Visualization")
    if len(regression_features) >= 2:
        fig8 = px.scatter_3d(df_reg, x=regression_features[0], y=regression_features[1], z=target_reg,
                             color=target_reg, title="3D: Risk Factor Regression")
        st.plotly_chart(fig8, use_container_width=True)
        st.caption("**Exec:** Where are high-incident employees on the feature spectrum?")

    # 9. Feature Correlation Heatmap
    st.subheader("9. Correlation Between Key Features")
    corr = df_emp_ml[['Score','IncidentsCaused','TrainingCount','RoleIdx']].corr()
    fig9 = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")
    st.plotly_chart(fig9, use_container_width=True)
    st.caption("**Exec:** Spot hidden relationships; anti-correlation between awareness & incidents is ideal.")

    # 10. ML-Driven Executive Recommendations
    st.subheader("10. AI-Generated Executive Action Points")
    # (For now, statically recommend top drivers—AI copilot could be added in future.)
    if "RF" in models and models["RF"][3] is not None:
        fi = pd.Series(models["RF"][3], index=X_cols)
        key_lever = fi.idxmax()
        st.info(f"**Recommendation:** Focus investments on improving '{key_lever}' to reduce severity of future incidents.")
# ========== TAB 10: ROOT CAUSE & DEEP DIVE EXPLORER ==========
with tabs[9]:
    st.header("Root Cause & Deep Dive Explorer")
    st.markdown("""
    _For CISO, CTO, and investigation teams: Analyze incident timelines, root causes, response speed, playbook effectiveness, and identify what must change. Every chart below is actionable for learning and board reporting._
    """)

    # 1. Incident Timeline (select type)
    st.subheader("1. Incident Timeline by Type")
    timeline_type = st.selectbox("Incident Type", df_incidents['IncidentType'].unique())
    timeline_data = df_incidents[df_incidents['IncidentType'] == timeline_type]
    fig1 = px.scatter(timeline_data, x='DateReported', y='Severity', color='Status', title=f"Timeline: {timeline_type}")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("**Exec:** Patterns of repeated incidents = poor fix or deeper root cause.")

    # 2. Incident Timeline (select employee)
    st.subheader("2. Incident Timeline by Employee")
    emp_names = df_emp['Name'].tolist()
    emp_select = st.selectbox("Employee", emp_names)
    emp_data = df_incidents[df_incidents['EmployeeID'] == df_emp[df_emp['Name'] == emp_select]['EmployeeID'].iloc[0]]
    fig2 = px.scatter(emp_data, x='DateReported', y='Severity', color='Status', title=f"Timeline: {emp_select}")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**HR:** Bursts for individuals often mean stress, new role, or training gap.")

    # 3. Root Cause Analysis: Top Patterns
    st.subheader("3. Top Root Causes")
    if 'RootCause' in df_incidents.columns:
        rc_counts = df_incidents['RootCause'].value_counts().reset_index()
        rc_counts.columns = ['RootCause','Count']
        fig3 = px.bar(rc_counts, x='RootCause', y='Count', title="Most Common Root Causes")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("**For RCA board:** Top 1-2 causes should get a dedicated action plan.")

    # 4. RCA Heatmap: Root Cause vs Department
    st.subheader("4. Root Cause by Department")
    if 'RootCause' in df_incidents.columns:
        rca_heat = pd.crosstab(df_incidents['RootCause'], 
                               df_incidents['DepartmentID'].map(df_dept.set_index('DepartmentID')['DepartmentName']))
        fig4 = px.imshow(rca_heat, text_auto=True, aspect="auto", title="Root Cause vs Department")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("**CxO:** Identify chronic/process causes by department.")

    # 5. Mean Time to Detect vs Respond
    st.subheader("5. Detection vs Response Lag")
    if 'DateResolved' in df_incidents.columns:
        df_incidents['DateResolved'] = pd.to_datetime(df_incidents['DateResolved'], errors='coerce')
        df_incidents['DateReported'] = pd.to_datetime(df_incidents['DateReported'])
        if 'DateDetected' in df_incidents.columns:
            df_incidents['DateDetected'] = pd.to_datetime(df_incidents['DateDetected'])
            df_incidents['DetectionLag'] = (df_incidents['DateDetected'] - df_incidents['DateReported']).dt.days
            df_incidents['ResponseLag'] = (df_incidents['DateResolved'] - df_incidents['DateDetected']).dt.days
            fig5 = px.box(df_incidents, y=['DetectionLag','ResponseLag'], points="all", title="Lag Days: Detection vs Response")
            st.plotly_chart(fig5, use_container_width=True)
            st.caption("**Board:** Long detection lags = missed monitoring; long response = resourcing or workflow gap.")

    # 6. Incident Outlier Analysis
    st.subheader("6. Outlier Incidents (by Duration or Loss)")
    if 'EstimatedLoss' in df_incidents.columns:
        outlier = df_incidents[df_incidents['EstimatedLoss'] > df_incidents['EstimatedLoss'].quantile(0.95)]
        st.dataframe(outlier[['IncidentID','IncidentType','Severity','EstimatedLoss','DateReported','Status']], use_container_width=True)
        st.caption("**Board:** Outliers may be root-cause examples for board or external review.")

    # 7. Playbook Used per Incident
    st.subheader("7. Incident Response Playbook Utilization")
    if 'Playbook' in df_incidents.columns:
        playbook_counts = df_incidents['Playbook'].value_counts().reset_index()
        playbook_counts.columns = ['Playbook','Count']
        fig7 = px.bar(playbook_counts, x='Playbook', y='Count', title="Playbook Usage Frequency")
        st.plotly_chart(fig7, use_container_width=True)
        st.caption("**Ops:** Over-used playbooks may be default, not optimal. Under-used ones may need awareness.")

    # 8. Lessons Learned Summary Table
    st.subheader("8. Lessons Learned Log")
    if 'LessonsLearned' in df_incidents.columns:
        lessons = df_incidents[df_incidents['LessonsLearned'].notnull()][['IncidentID','LessonsLearned']]
        st.dataframe(lessons, use_container_width=True)
        st.caption("**CISO:** Regular review of lessons learned = maturing security culture.")

    # 9. 3D Root Cause View
    st.subheader("9. 3D Incident Root Cause Mapping")
    if 'RootCause' in df_incidents.columns:
        rc_map = {k: i for i, k in enumerate(df_incidents['RootCause'].dropna().unique())}
        df_3drc = df_incidents.dropna(subset=['RootCause']).copy()
        df_3drc['RCIdx'] = df_3drc['RootCause'].map(rc_map)
        fig9 = px.scatter_3d(df_3drc, x='RCIdx', y='Severity', z='EstimatedLoss' if 'EstimatedLoss' in df_3drc.columns else 'AIResponseTime',
                             color='RootCause', hover_data=['IncidentID','DateReported'],
                             title="3D: Root Cause, Severity, Loss/Response")
        st.plotly_chart(fig9, use_container_width=True)
        st.caption("**CxO:** Spot root causes that drive big loss or high severity.")

    # 10. Executive Action Plan Generator (based on findings)
    st.subheader("10. Executive Action Plan")
    top_cause = rc_counts['RootCause'].iloc[0] if 'rc_counts' in locals() and not rc_counts.empty else "Unknown"
    st.info(f"**Recommended Next Action:** Focus next quarter's effort on root cause '{top_cause}'. Build a campaign for awareness, new controls, or leadership review.")
