import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import plotly.express as px

def kmeans_cluster(df, features, n_clusters=3):
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, cluster_labels)
    df_clustered = df.copy()
    df_clustered['cluster'] = -1
    df_clustered.loc[X.index, 'cluster'] = cluster_labels
    return df_clustered, kmeans, sil_score

def cluster_plot(df, feature_x, feature_y):
    fig = px.scatter(df, x=feature_x, y=feature_y, color='cluster',
                     title=f"K-means Clustering ({feature_x} vs {feature_y})")
    return fig

def severity_classifier(df, features, target='Severity'):
    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = LabelEncoder().fit_transform(df_clean[target])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    return model, feature_importances

def regression_analysis(df, features, target):
    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]
    model = LinearRegression()
    model.fit(X, y)
    coef = model.coef_
    return model, coef
