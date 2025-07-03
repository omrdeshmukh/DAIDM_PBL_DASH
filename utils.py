import pandas as pd
import os

def load_data(datapath='data'):
    files = ['employees.csv', 'departments.csv', 'roles.csv', 'training.csv',
             'incidents.csv', 'security_events.csv', 'agentic_ai_log.csv', 
             'blockchain_audit_log.csv', 'policy.csv']
    dfs = []
    for f in files:
        path = os.path.join(datapath, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        dfs.append(pd.read_csv(path))
    return dfs
