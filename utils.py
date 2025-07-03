import pandas as pd

def load_data():
    df_emp = pd.read_csv('data/employees.csv')
    df_dept = pd.read_csv('data/departments.csv')
    df_roles = pd.read_csv('data/roles.csv')
    df_training = pd.read_csv('data/training.csv')
    df_incidents = pd.read_csv('data/incidents.csv')
    df_events = pd.read_csv('data/security_events.csv')
    df_ai = pd.read_csv('data/agentic_ai_log.csv')
    df_bc = pd.read_csv('data/blockchain_audit_log.csv')
    df_policy = pd.read_csv('data/policy.csv')
    return df_emp, df_dept, df_roles, df_training, df_incidents, df_events, df_ai, df_bc, df_policy
