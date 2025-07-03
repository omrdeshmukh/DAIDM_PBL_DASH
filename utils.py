import pandas as pd

def load_data_from_upload(uploaded_files):
    """
    Accepts a list of uploaded Streamlit files, returns a dict of DataFrames keyed by filename (without .csv)
    """
    dfs = {}
    for file in uploaded_files:
        name = file.name.replace('.csv','')
        dfs[name] = pd.read_csv(file)
    return dfs
