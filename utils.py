import pandas as pd
import os

def load_data(datapath='data'):
    files = ['employees.csv', 'departments.csv', 'roles.csv', 'training.csv',
             'incidents.csv', 'security_events.csv', 'agentic_ai_log.csv',
             'blockchain_audit_log.csv', 'policy.csv']
    dfs = {}
    for f in files:
        path = os.path.join(datapath, f)
        if os.path.exists(path):
            dfs[f.replace('.csv','')] = pd.read_csv(path)
        else:
            dfs[f.replace('.csv','')] = None
    return dfs

def load_data_from_upload(uploaded_files):
    dfs = {}
    for file in uploaded_files:
        name = file.name.replace('.csv','')
        dfs[name] = pd.read_csv(file)
    return dfs
