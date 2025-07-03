import pandas as pd

def clean_incidents_data(df_incidents, df_emp, df_training):
    # Merge in employee info (always keep EmployeeID, DepartmentID, RoleID, Location)
    emp_feats = df_emp[['EmployeeID', 'DepartmentID', 'RoleID', 'Location']]
    df = df_incidents.merge(emp_feats, on='EmployeeID', how='left')
    # Merge in average awareness score
    awareness = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness'].groupby('EmployeeID')['Score'].mean().reset_index()
    df = df.merge(awareness, on='EmployeeID', how='left', suffixes=('','_awareness'))
    # Create categorical codes
    for col in ['DepartmentID', 'RoleID', 'Location', 'IncidentType']:
        if col in df.columns:
            df[f"{col}Idx"] = df[col].astype('category').cat.codes
    # Fill NA as needed
    df['Score'] = df['Score'].fillna(0)
    return df

def clean_employee_ml_data(df_emp, df_incidents, df_training):
    # All features: Score, IncidentsCaused, TrainingCount, RoleID
    emp_ml = df_emp[['EmployeeID', 'Name', 'DepartmentID', 'RoleID']].copy()
    emp_awareness = df_training[df_training['TrainingType'] == 'Cybersecurity Awareness'].groupby('EmployeeID')['Score'].mean().reset_index()
    emp_incidents = df_incidents.groupby('EmployeeID').size().reset_index(name='IncidentsCaused')
    emp_training_count = df_training.groupby('EmployeeID').size().reset_index(name='TrainingCount')
    emp_ml = emp_ml.merge(emp_awareness, on='EmployeeID', how='left')
    emp_ml = emp_ml.merge(emp_incidents, on='EmployeeID', how='left').fillna({'Score':0, 'IncidentsCaused':0})
    emp_ml = emp_ml.merge(emp_training_count, on='EmployeeID', how='left').fillna({'TrainingCount':0})
    emp_ml['RoleIdx'] = emp_ml['RoleID'].astype('category').cat.codes
    return emp_ml
