import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data, load_data_from_upload
from ml_models import (
    kmeans_cluster, get_elbow_curve, get_silhouette_scores, plot_elbow_curve, plot_silhouette_curve,
    cluster_3d_plot, train_classifiers, plot_confusion, train_regressors
)

st.set_page_config(page_title="CyberSOC-aaS Executive Dashboard", layout="wide")
st.title("Blockchain-Enabled Agentic AI Cybersecurity Platform")

# ------------- DATA LOAD LOGIC (supports both backend and upload) ---------------
# Sidebar - allow either backend load or upload
data_mode = st.sidebar.radio("Data Source", ["Backend CSVs", "Upload CSVs"], index=0)
dfs = {}
if data_mode == "Backend CSVs":
    dfs = load_data('data')
    missing_files = [k for k,v in dfs.items() if v is None]
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

# Unpack
df_emp = dfs['employees']
df_dept = dfs['departments']
df_roles = dfs['roles']
df_training = dfs['training']
df_incidents = dfs['incidents']
df_events = dfs['security_events']
df_ai = dfs['agentic_ai_log']
df_bc = dfs['blockchain_audit_log']
df_policy = dfs['policy']

# ---------- App Tabs ----------
tabs = st.tabs([
    "🏠 Executive Overview", 
    "📊 Cybersecurity Metrics", 
    "🤖 Agentic AI & Blockchain", 
    "🧠 ML & Predictive Analytics", 
    "🕵️ Employee/Asset Explorer"
])
# ------------- EXECUTIVE OVERVIEW TAB -------------
with tabs[0]:
    st.header("Executive Summary & KPIs")
    st.markdown("""
    _This dashboard offers C-suite executives a real-time pulse of the organization's cybersecurity readiness. 
    Instantly see risk trends, incident trends, and how well the workforce is prepared through continuous training and AI-driven protection._
    """)

    # Key numbers
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", df_emp.shape[0], help="Current size of organization being protected.")
    col2.metric("Total Incidents (1 yr)", df_incidents.shape[0], help="Number of unique reported security incidents in the last year.")
    col3.metric("Trainings Completed", (df_training['Status'] == "Completed").sum(), help="Sum of all cybersecurity training completions by staff.")
    col4.metric("Resolved Incidents", (df_incidents['Status'] == "Closed").sum(), help="Security incidents marked resolved by AI, IT, or both.")

    st.subheader("🧑‍💻 Workforce Cyber Awareness")
    st.write("**Distribution of Cybersecurity Awareness Scores.** A right-skewed distribution means effective upskilling. Spikes to the left suggest more training needed.")
    awareness = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness']
    fig_awareness = px.histogram(awareness, x="Score", nbins=25, title="Employee Awareness Scores", color="Status")
    st.plotly_chart(fig_awareness, use_container_width=True)

    st.subheader("📈 Incident Trends")
    st.write("**Incident Frequency Over Time.** Spikes may correlate with industry threats, employee churn, or infrastructure changes. Use this for strategic board updates.")
    df_incidents['DateReported'] = pd.to_datetime(df_incidents['DateReported'])
    trend = df_incidents.groupby(df_incidents['DateReported'].dt.to_period('M')).size()
    fig_trend = px.line(
        x=trend.index.astype(str), y=trend.values, markers=True,
        title="Monthly Security Incident Reports", labels={'x': 'Month', 'y': 'Number of Incidents'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("📝 Active Compliance Policies")
    st.write("**Current security policies, version control, and enforcement ownership (AI/Human/Hybrid).** Useful for audits and board-level compliance attestation.")
    st.dataframe(df_policy.sort_values('DateActive', ascending=False).head(8), use_container_width=True)

# ------------- CYBERSECURITY METRICS TAB -------------
with tabs[1]:
    st.header("Operational Cybersecurity Metrics")
    st.markdown("""
    _These metrics help the security team, CTO, and risk managers identify weak spots and high-risk departments at a glance. 
    Use these for weekly risk reviews, resource allocation, and budget planning._
    """)
    
    st.subheader("Incident Type Distribution")
    st.write("Visualizes what types of attacks are most common (e.g., phishing, malware, data leak). Spike in any type signals a need for more training or technical controls.")
    fig_types = px.pie(df_incidents, names="IncidentType", title="Incidents by Type")
    st.plotly_chart(fig_types, use_container_width=True)
    
    st.subheader("Incident Severity by Department")
    st.write("Heatmap shows which departments are targeted (or compromised) most often, and at what severity. Use this to guide targeted training or controls.")
    dept_sev = pd.crosstab(df_incidents['DepartmentID'], df_incidents['Severity'])
    dept_sev.index = dept_sev.index.map(df_dept.set_index('DepartmentID')['DepartmentName'].to_dict())
    fig_heat = px.imshow(dept_sev, text_auto=True, aspect="auto", title="Severity by Department")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.subheader("Top Risk Departments")
    st.write("Bar chart of departments by incident count. Persistent top-3 risk areas suggest structural process gaps or high turnover.")
    top_depts = df_incidents['DepartmentID'].value_counts().rename(index=df_dept.set_index('DepartmentID')['DepartmentName'].to_dict())
    fig_bar = px.bar(x=top_depts.index, y=top_depts.values, labels={'x':'Department','y':'Incident Count'}, title="Incident Volume by Department")
    st.plotly_chart(fig_bar, use_container_width=True)
# ------------- AGENTIC AI & BLOCKCHAIN TAB -------------
with tabs[2]:
    st.header("Agentic AI Automation & Blockchain Audit")
    st.markdown("""
    _This section gives executives and security leads a live log of what the AI is doing (auto-response, confidence, human escalations), 
    plus evidence of all actions being immutably recorded on the blockchain. 
    This transparency is crucial for compliance, regulator trust, and cyber insurance negotiations._
    """)

    st.subheader("Latest Agentic AI Actions")
    st.write("Displays a rolling log of the most recent AI-driven responses: what action, what agent, confidence score, and whether it was escalated to a human analyst.")
    st.dataframe(
        df_ai[['Timestamp','AgentID','ActionTaken','DecisionConfidence','EscalatedToHuman','Outcome']]
        .sort_values('Timestamp', ascending=False).head(30), use_container_width=True
    )

    st.subheader("AI Decision Confidence by Action")
    st.write("Box plot shows how confident the AI is for each type of action. Wide range = more human review needed. Narrow/high = strong AI reliability.")
    fig_conf = px.box(df_ai, x='ActionTaken', y='DecisionConfidence', title="AI Decision Confidence by Action")
    st.plotly_chart(fig_conf, use_container_width=True)

    st.subheader("Blockchain Audit Log")
    st.write("Sample of blockchain-backed log entries: every critical event/action is stored immutably and is instantly auditable. Useful for demonstrating compliance to regulators or auditors.")
    st.dataframe(df_bc.sample(10), use_container_width=True)

# ------------- EMPLOYEE/ASSET EXPLORER TAB -------------
with tabs[4]:
    st.header("Employee/Asset Risk Explorer")
    st.markdown("""
    _Drill down into any individual (or device) to see all their training, incidents, and events. 
    This helps HR, CISO, and IT operations to investigate insider threats, habitual offenders, or rising stars in cyber hygiene._
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
# ------------- ML & PREDICTIVE ANALYTICS TAB -------------
with tabs[3]:
    st.header("ML & Predictive Analytics")
    st.markdown("""
    _This section empowers CTOs, CISOs, and analysts to discover risk patterns, automate predictions, and measure the impact of cyber investments on the business. 
    Each ML output is explained for execs, so findings can inform both technical and strategic decisions._
    """)

    st.subheader("K-Means Clustering (with Elbow, Silhouette, and 3D Plots)")
    st.write("""
    **Clustering reveals natural risk groups among employees—outliers often indicate persistent risk or superstar performers.  
    Use the elbow and silhouette methods to choose the right number of clusters. 3D plots make it easy to spot risk patterns.**
    """)

    # Prepare features (example: awareness + incidents + training count)
    df_emp_ml = df_emp.copy()
    emp_awareness = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness'].groupby('EmployeeID')['Score'].mean().reset_index()
    emp_incidents = df_incidents.groupby('EmployeeID').size().reset_index(name='IncidentsCaused')
    emp_training_count = df_training.groupby('EmployeeID').size().reset_index(name='TrainingCount')
    df_emp_ml = pd.merge(df_emp_ml, emp_awareness, on='EmployeeID', how='left')
    df_emp_ml = pd.merge(df_emp_ml, emp_incidents, on='EmployeeID', how='left').fillna({'Score':0, 'IncidentsCaused':0})
    df_emp_ml = pd.merge(df_emp_ml, emp_training_count, on='EmployeeID', how='left').fillna({'TrainingCount':0})

    cluster_features = st.multiselect("Clustering Features", ['Score', 'IncidentsCaused', 'TrainingCount'], default=['Score','IncidentsCaused'])
    max_k = st.slider("Elbow/Silhouette: Max clusters (K)", min_value=3, max_value=10, value=6)
    if len(cluster_features) >= 2:
        # Clean features
        X = df_emp_ml[cluster_features].dropna().to_numpy()
        inertia = get_elbow_curve(X, max_k=max_k)
        sil_scores = get_silhouette_scores(X, max_k=max_k)
        st.plotly_chart(plot_elbow_curve(inertia, max_k=max_k), use_container_width=True)
        st.plotly_chart(plot_silhouette_curve(sil_scores, max_k=max_k), use_container_width=True)

        n_clusters = st.slider("Number of Clusters to Display", min_value=2, max_value=max_k, value=3)
        df_clustered, kmeans, sil_score = kmeans_cluster(df_emp_ml, cluster_features, n_clusters)
        st.success(f"Silhouette Score for {n_clusters} clusters: {sil_score:.3f} (Higher = better defined clusters)")
        st.write("**2D Cluster Plot**")
        st.plotly_chart(px.scatter(df_clustered, x=cluster_features[0], y=cluster_features[1], color='cluster', title="K-means Clusters"), use_container_width=True)
        if len(cluster_features) >= 3:
            st.write("**3D Cluster Plot**")
            st.plotly_chart(cluster_3d_plot(df_clustered, cluster_features[0], cluster_features[1], cluster_features[2]), use_container_width=True)
        st.write("**Business interpretation:** Outlier clusters represent pockets of persistent risk (low awareness, high incidents), or potential role models.")

    st.divider()

    # --- Classification Section ---
    st.subheader("Incident Severity Prediction (Multiple Classifiers)")
    st.write("""
    **How well can we predict which incidents will be severe?**  
    Try multiple ML models and compare. Confusion matrix shows what the AI "gets right" vs. where it struggles.  
    Feature importances reveal which factors matter most.
    """)
    # Prepare data for classification
    df_class = df_incidents.copy()
    feat_map = {k: v for v, k in enumerate(df_dept['DepartmentName'].values)}
    df_class['DeptIdx'] = df_class['DepartmentID'].map(feat_map)
    df_class['TypeIdx'] = df_class['IncidentType'].astype('category').cat.codes
    # Ensure all features and target are numeric, clean NaNs/infs
    X_cols = ['AIResponseTime', 'DeptIdx', 'TypeIdx']
    for col in X_cols:
        df_class[col] = pd.to_numeric(df_class[col], errors='coerce')
    df_class = df_class.dropna(subset=X_cols + ['Severity'])
    target_label = 'Severity'
    if df_class.empty:
        st.warning("Not enough clean data for classification. Please check for missing AIResponseTime or Severity values.")
    else:
        models, y_test = train_classifiers(df_class, X_cols, target_label)
        st.write("**Model Comparison:**")
        for name, (model, score, cm) in models.items():
            st.write(f"**{name}**: Test accuracy = {score:.2%}")
            st.plotly_chart(plot_confusion(cm, labels=np.unique(df_class[target_label]), title=f"{name} Confusion Matrix"), use_container_width=True)
        st.write("""
        **Interpretation:** Higher accuracy means better incident risk prediction.  
        Confusion matrices show where models misclassify severity (e.g., calling a 'High' event 'Medium').  
        For board risk reports, focus on reducing severe-class false negatives.
        """)

    st.divider()

    # --- Regression Section ---
    st.subheader("Incident Cause Risk Prediction (Regression)")
    st.write("""
    **Which factors drive the number of incidents an employee causes?**  
    Run multiple regression models to understand which levers (awareness, training, role) have most impact.
    """)
    df_reg = df_emp_ml.copy()
    df_reg['RoleIdx'] = df_reg['RoleID']
    regression_features = st.multiselect("Regression Features", ['Score', 'RoleIdx', 'TrainingCount'], default=['Score','RoleIdx'])
    target_reg = 'IncidentsCaused'
    if df_reg[regression_features + [target_reg]].dropna().empty:
        st.warning("Not enough clean data for regression. Please select valid features.")
    else:
        regressors = train_regressors(df_reg, regression_features, target_reg)
        for name, (model, r2, coef) in regressors.items():
            st.write(f"**{name} Regression**: R2 = {r2:.2f}")
            if coef is not None:
                st.bar_chart(pd.Series(coef, index=regression_features))
        st.write("""
        **Interpretation:** R2 tells you how well the model explains who will cause more incidents.  
        The bar chart shows which factors matter most. Use this to design targeted training or access reviews.
        """)

    st.info("""
    _Every ML chart here is executive-ready.  
    CEOs and CTOs can see at-a-glance which segments are most at risk, where AI/ML is succeeding or struggling, 
    and which interventions will yield the highest risk reduction per dollar spent._
    """)
