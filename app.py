# app.py — Segmentación de clientes con K-Means (Streamlit)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Segmentación de clientes", page_icon="🧩", layout="centered")
st.title("Segmentación de clientes con K-Means / Axel Mireles")

st.write(
    "La app normaliza (0–1) con **MinMaxScaler**, aplica **K-Means** y grafica los clústeres. "
    "Puedes subir tu CSV o usar **clientes.csv** del repositorio."
)

# ---------------------------
# 1) Lectura de datos
# ---------------------------
archivo = st.file_uploader("Sube un CSV (opcional)", type=["csv"])

if archivo is None:
    # CSV por defecto en el repo
    df = pd.read_csv("clientes.csv")
    st.info("Usando el archivo por defecto: `clientes.csv`")
else:
    df = pd.read_csv(archivo)
    st.success("Archivo cargado correctamente.")

st.subheader("Vista previa de datos")
st.dataframe(df.head())

# ---------------------------
# 2) Selección de columnas
# ---------------------------
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(num_cols) < 2:
    st.error("Se requieren **al menos dos columnas numéricas**.")
    st.stop()

# Intentar usar nombres típicos si existen
def find_col(name_normalizado: str):
    for c in df.columns:
        if c.strip().lower() == name_normalizado:
            return c
    return None

default_x = find_col("saldo") or num_cols[0]
default_y = find_col("transacciones") or (num_cols[1] if len(num_cols) > 1 else num_cols[0])

st.sidebar.header("Parámetros")
x_col = st.sidebar.selectbox("Columna X", options=num_cols, index=num_cols.index(default_x))
y_col = st.sidebar.selectbox("Columna Y", options=num_cols, index=num_cols.index(default_y))
k = st.sidebar.slider("Número de clústeres (k)", min_value=2, max_value=8, value=3, step=1)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

X = df[[x_col, y_col]].copy()

# ---------------------------
# 3) Escalado (0–1)
# ---------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.values)

# ---------------------------
# 4) Método del codo (opcional)
# ---------------------------
with st.expander("Ver método del codo (inercia vs k)"):
    ks = list(range(2, 11))
    inercias = []
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=random_state)
        km.fit(X_scaled)
        inercias.append(km.inertia_)
    fig_elbow, ax_elbow = plt.subplots(figsize=(5, 4))
    ax_elbow.plot(ks, inercias, marker="o")
    ax_elbow.set_xlabel("Número de clústeres (k)")
    ax_elbow.set_ylabel("Inercia")
    ax_elbow.set_title("Método del codo")
    st.pyplot(fig_elbow)

# ---------------------------
# 5) K-Means final
# ---------------------------
kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
labels = kmeans.fit_predict(X_scaled)

# Centroides (volver a escala original)
cent_scaled = kmeans.cluster_centers_
cent_original = scaler.inverse_transform(cent_scaled)
centros_df = pd.DataFrame(cent_original, columns=[x_col, y_col])
centros_df.index.name = "cluster"

df_result = df.copy()
df_result["cluster"] = labels

st.subheader("Centroides (escala original)")
st.dataframe(centros_df)

# ---------------------------
# 6) Gráfica (escala original)
# ---------------------------
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["red", "blue", "orange", "black", "purple", "pink", "brown", "green"]
for c in range(k):
    m = (labels == c)
    ax.scatter(X[m][x_col], X[m][y_col], s=80, alpha=0.65, label=f"cluster {c}", color=colors[c % len(colors)])
ax.scatter(centros_df[x_col], centros_df[y_col], s=220, marker="P", color="black", label="centroides")
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title("Clústeres (escala original)")
ax.legend()
st.pyplot(fig)

# ---------------------------
# 7) Descargar CSV con clústeres
# ---------------------------
st.subheader("Descargar resultados")
st.download_button(
    "Descargar CSV con clústeres",
    data=df_result.to_csv(index=False).encode("utf-8"),
    file_name="clientes_segmentados.csv",
    mime="text/csv",
)
