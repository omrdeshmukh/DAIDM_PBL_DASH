import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import plotly.express as px
import plotly.graph_objs as go

def get_elbow_curve(X, max_k=10):
    inertia = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    return inertia

def get_silhouette_scores(X, max_k=10):
    sil_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
    return sil_scores

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

def plot_elbow_curve(inertia, max_k=10):
    fig = px.line(x=list(range(1, max_k+1)), y=inertia, markers=True,
                  title="Elbow Method: Optimal Number of K-Means Clusters",
                  labels={"x": "Number of Clusters (k)", "y": "Inertia (Distortion)"})
    return fig

def plot_silhouette_curve(scores, max_k=10):
    fig = px.line(x=list(range(2, max_k+1)), y=scores, markers=True,
                  title="Silhouette Score vs. Number of Clusters",
                  labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"})
    return fig

def cluster_3d_plot(df, x, y, z):
    fig = px.scatter_3d(df, x=x, y=y, z=z, color='cluster',
                        title=f"3D K-means Clusters: {x} vs {y} vs {z}",
                        custom_data=['EmployeeID', 'Name', 'DepartmentID', 'RoleID'])
    fig.update_traces(
        hovertemplate="Employee: %{customdata[1]}<br>"+\
                      f"{x}: "+"%{x}<br>"+f"{y}: "+"%{y}<br>"+f"{z}: "+"%{z}<br>"+
                      "Dept: %{customdata[2]}<br>Role: %{customdata[3]}"
    )
    return fig

# --- Enhanced Classifiers ---
def train_classifiers(df, features, target, label_encoder=None):
    from sklearn.model_selection import train_test_split
    results = {}
    df = df.dropna(subset=features + [target])
    X = df[features]
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[target])
    else:
        y = label_encoder.transform(df[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    results['KNN'] = (knn, knn.score(X_test, y_test), confusion_matrix(y_test, knn.predict(X_test)))

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    results['DT'] = (dt, dt.score(X_test, y_test), confusion_matrix(y_test, dt.predict(X_test)))

    # Random Forest (with class weights)
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    results['RF'] = (rf, rf.score(X_test, y_test), confusion_matrix(y_test, rf.predict(X_test)), rf.feature_importances_)

    # Gradient Boosting
    gbt = GradientBoostingClassifier(random_state=42)
    gbt.fit(X_train, y_train)
    results['GBRT'] = (gbt, gbt.score(X_test, y_test), confusion_matrix(y_test, gbt.predict(X_test)))

    return results, y_test, label_encoder

def plot_confusion(cm, labels, title="Confusion Matrix"):
    fig = px.imshow(cm, text_auto=True, title=title, labels=dict(x="Predicted", y="Actual"))
    fig.update_xaxes(tickvals=list(range(len(labels))), ticktext=labels)
    fig.update_yaxes(tickvals=list(range(len(labels))), ticktext=labels)
    return fig

# --- Enhanced Regressors ---
def train_regressors(df, features, target):
    from sklearn.model_selection import train_test_split
    results = {}
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results['Linear'] = (lr, lr.score(X_test, y_test), lr.coef_)

    # Ridge Regression
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    results['Ridge'] = (ridge, ridge.score(X_test, y_test), ridge.coef_)

    # Lasso Regression
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    results['Lasso'] = (lasso, lasso.score(X_test, y_test), lasso.coef_)

    # Decision Tree Regression
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    results['DT'] = (dt, dt.score(X_test, y_test), None)

    # Random Forest Regression
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results['RF'] = (rf, rf.score(X_test, y_test), rf.feature_importances_)

    return results
