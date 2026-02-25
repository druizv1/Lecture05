import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configuración de la página
st.set_page_config(page_title="Iris Classifier Explorer", layout="wide")

st.title("🌸 Explorador de Clasificación: Dataset Iris")
st.markdown("""
Esta app permite ajustar modelos de Machine Learning y visualizar cómo separan las categorías de flores.
""")

# --- SIDEBAR: CONFIGURACIÓN ---
st.sidebar.header("Configuración del Modelo")

classifier_name = st.sidebar.selectbox(
    "Selecciona el Clasificador",
    ("SVM", "Random Forest", "Logistic Regression")
)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C (Regularización)", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth", 1, 15, 5)
        n_estimators = st.sidebar.slider("N Estimators", 1, 100, 10)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    elif clf_name == "Logistic Regression":
        C = st.sidebar.slider("C (Inversa de regularización)", 0.01, 10.0)
        params["C"] = C
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "SVM":
        clf = SVC(C=params["C"], probability=True)
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                    max_depth=params["max_depth"], random_state=42)
    else:
        clf = LogisticRegression(C=params["C"], max_iter=1000)
    return clf

# --- CARGA DE DATOS ---
iris = datasets.load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# --- ENTRENAMIENTO ---
clf = get_classifier(classifier_name, params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# --- LAYOUT DE RESULTADOS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Métricas de {classifier_name}")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")
    
    # Mostrar reporte en tabla
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose().drop(columns="support"))

    # Matriz de Confusión
    st.write("**Matriz de Confusión:**")
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    st.pyplot(fig_cm)

with col2:
    st.subheader("Fronteras de Decisión (PCA)")
    st.info("Nota: Reducimos a 2 dimensiones para visualizar la frontera.")
    
    # Reducción de dimensionalidad para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Re-entrenar modelo con 2D para el gráfico
    clf_2d = get_classifier(classifier_name, params)
    clf_2d.fit(X_pca, y)

    # Crear malla (Meshgrid)
    h = .02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig_dec, ax_dec = plt.subplots()
    ax_dec.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = ax_dec.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap='viridis')
    ax_dec.set_xlabel('Componente Principal 1')
    ax_dec.set_ylabel('Componente Principal 2')
    legend1 = ax_dec.legend(*scatter.legend_elements(), title="Clases")
    ax_dec.add_artist(legend1)
    
    st.pyplot(fig_dec)

# --- VISUALIZACIÓN DE DATOS ---
if st.checkbox("Mostrar datos crudos"):
    st.write(df)
