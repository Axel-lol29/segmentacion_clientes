# app.py — Aprendizaje no supervisado: k-means (Streamlit)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="K-Means", page_icon="🧩", layout="centered")

# ======= Encabezado (estilo similar a tu amigo) =======
st.title("Aprendizaje no supervisado: k-means")
st.subheader("Axel Mireles")

st.markdown("### cargar datos")
archivo = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

# --------- Lectura de datos ---------
if archivo is None:
    df = pd.read_csv("clientes.csv")
    st.info("Usando el archivo por defecto: **clientes.csv**")
else:
    df = pd.read_csv(archivo)

# --------- Detección de columnas numéricas y defaults amigables ---------
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(num_cols) < 2:
    st.error("Tu CSV debe tener **al menos dos columnas numéricas**.")
    st.stop()

def pick_col(name_lower: str, fallback_idx: int):
    # intenta matchear por nombre (ignorando mayúsculas/acentos simples comunes)
    for c in df.columns:
        if c.strip().lower() == name_lower:
            return c
    return num_cols[fallback_idx]

# Soporta nombres típicos: ingresos/puntuacion o saldo/transacciones
x_def = pick_col("ingresos", 0)
y_def = pick_col("puntuacion", 1 if len(num_cols) > 1 else 0)
x_def = pick_col("saldo", num_cols.index(x_def))         # si existe "saldo", úsalo
y_def = pick_col("transacciones", num_cols.index(y_def)) # si existe "transacciones", úsalo

# --------- Vista de datos (como en tu amigo) ---------
st.markdown("### Datos")
st.dataframe(df[[x_def, y_def]].head(10), use_container_width=True)

# --------- Normalización 0–1 y tabla normalizada ---------
X = df[[x_def, y_def]].copy()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.values)
df_scaled = pd.DataFrame(X_scaled, columns=["Saldo", "transacciones"])  # nombres como en su ejemplo

st.dataframe(df_scaled.head(10), use_container_width=True)

# --------- Parámetros del modelo ---------
k = st.sidebar.slider("k (número de clústeres)", 2, 9, 3, 1)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# --------- Entrenamiento K-Means ---------
kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
labels = kmeans.fit_predict(X_scaled)

# Centroides (en escala normalizada y en escala original)
cent_scaled = kmeans.cluster_centers_
cent_original = scaler.inverse_transform(cent_scaled)

# Mostrar centroides (como líneas de texto, estilo del ejemplo)
st.write(cent_scaled.tolist())
st.write(kmeans.inertia_)

# --------- Gráfica scatter Matplotlib (estilo simple como su app) ---------
fig, ax = plt.subplots(figsize=(6, 5))
# colores fijos similares
colors = ["#FF6B6B", "#4D96FF", "#FFB86B", "#6BCB77", "#C77DFF", "#FFD166", "#00C2A8", "#9B59B6", "#2ECC71"]

for c in range(k):
    m = (labels == c)
    ax.scatter(df_scaled.iloc[m, 0], df_scaled.iloc[m, 1], s=60, alpha=0.9, color=colors[c % len(colors)])

ax.set_title("clientes")
ax.set_xlabel("saldo en cuenta de ahorros")
ax.set_ylabel("veces que uso tarjeta de credito")

# Textos laterales como en su captura
ax.text(1.02, 0.85, f"k={k}", transform=ax.transAxes)
ax.text(1.02, 0.78, f"inercia = {kmeans.inertia_:.2f}", transform=ax.transAxes)

st.pyplot(fig)

# --------- Método del codo ---------
ks = list(range(2, 10))
inercias = []
for kk in ks:
    km = KMeans(n_clusters=kk, n_init=10, random_state=random_state)
    km.fit(X_scaled)
    inercias.append(km.inertia_)

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.scatter(ks, inercias, s=60, color="#7E57C2")
ax2.plot(ks, inercias, color="#7E57C2")
ax2.set_xlabel("numero de clusters")
ax2.set_ylabel("inercia")
st.pyplot(fig2)

# --------- Resultado con etiquetas por si quieres descargar ---------
df_result = df.copy()
df_result["cluster"] = labels
st.download_button(
    "Descargar CSV con clústeres",
    data=df_result.to_csv(index=False).encode("utf-8"),
    file_name="clientes_segmentados.csv",
    mime="text/csv",
)
