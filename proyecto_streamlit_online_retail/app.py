from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Segmentación de clientes RFM", layout="wide")
BASE = Path(__file__).resolve().parent

# CAPTURA A — Carga del modelo entrenado y datos históricos
@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "modelo.joblib")

@st.cache_data
def cargar_tablas():
    return (pd.read_csv(BASE / "clientes.csv", dtype={"CustomerID": str}),
            pd.read_csv(BASE / "perfiles.csv"), pd.read_csv(BASE / "metricas.csv"))

# CAPTURA B — Preprocesamiento e inferencia sin reentrenamiento
def asignar(bundle, valores):
    fila = pd.DataFrame([valores], columns=bundle["features"])
    x = bundle["scaler"].transform(np.log1p(fila))
    cluster = int(bundle["model"].predict(x)[0])
    punto = bundle["pca"].transform(x)[0]
    distancia = float(bundle["model"].transform(x)[0, cluster])
    return cluster, punto, distancia

try:
    bundle = cargar_modelo()
    clientes, perfiles, metricas = cargar_tablas()
except (FileNotFoundError, ValueError) as e:
    st.error("No se pudieron cargar los artefactos. Ejecuta el cuaderno completo y coloca sus archivos junto a app.py.")
    st.stop()

st.title("Segmentación de clientes de comercio electrónico")
st.write("Consulta el comportamiento histórico de un cliente o ingresa sus valores RFM para identificar el segmento más cercano.")
st.caption(f"Online Retail · {len(clientes):,} clientes · K-Means k={bundle['model'].n_clusters} · Fecha de referencia histórica: {bundle['reference_date']}")

# CAPTURA C — Formulario y selección de un registro existente
modo = st.radio("Modo de consulta", ["Cliente existente", "Ingresar valores RFM"], horizontal=True)
if modo == "Cliente existente":
    cid = st.selectbox("CustomerID", clientes["CustomerID"].tolist())
    fila = clientes.loc[clientes["CustomerID"] == cid].iloc[0]
    valores = fila[bundle["features"]].astype(float).tolist()
    st.dataframe(pd.DataFrame([dict(zip(bundle["features"], valores))]), hide_index=True)
    mostrar = True
else:
    st.write("Usa una ventana de observación comparable al año del dataset. Monetary se expresa en libras esterlinas y Frequency cuenta facturas distintas.")
    with st.form("rfm"):
        r = st.number_input("Recency — días desde la última compra", min_value=0, value=30, step=1)
        f = st.number_input("Frequency — facturas distintas", min_value=1, value=3, step=1)
        m = st.number_input("Monetary — gasto acumulado en GBP", min_value=0.01, value=1000.0, step=10.0)
        mostrar = st.form_submit_button("Asignar segmento")
    valores = [r, f, m]

# CAPTURA D — Resultado, perfil y visualización de la consulta
if mostrar:
    if not np.isfinite(valores).all():
        st.error("Ingresa valores numéricos finitos.")
        st.stop()
    fuera = [v for v, x in zip(bundle["features"], valores)
             if x < bundle["min"][v] or x > bundle["max"][v]]
    if fuera:
        st.warning("Valores fuera del rango histórico: " + ", ".join(fuera) + ". La asignación requiere cautela.")
    cluster, punto, distancia = asignar(bundle, valores)
    st.success(f"Cluster {cluster} — {bundle['names'][cluster]}")
    st.caption(f"Distancia al centroide en el espacio estandarizado: {distancia:.3f}. No representa una probabilidad.")
    st.subheader("Perfil típico del segmento")
    st.dataframe(perfiles[perfiles["Cluster"] == cluster], hide_index=True)
    st.subheader("Ubicación del cliente en los segmentos")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for c, grupo in clientes.groupby("Cluster"):
        ax.scatter(grupo["PC1"], grupo["PC2"], s=10, alpha=.3, label=bundle["names"][int(c)])
    ax.scatter(punto[0], punto[1], marker="*", c="black", s=220, edgecolors="white", label="Cliente consultado")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"PCA conserva {bundle['pca'].explained_variance_ratio_.sum():.1%} de la varianza. El cluster se calcula en las tres variables RFM.")

st.subheader("Evaluación de los modelos")
st.dataframe(metricas.round(4), hide_index=True)
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, col in zip(axes, ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"]):
    ax.bar(metricas["Modelo"], metricas[col], color=["#2E75B6", "#70AD47"])
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Mayor Silhouette y Calinski-Harabasz, y menor Davies-Bouldin, favorecen la calidad interna. La comparación es exploratoria sobre el dataset histórico.")
st.subheader("Perfiles de todos los segmentos")
st.dataframe(perfiles, hide_index=True)
st.download_button("Descargar clientes segmentados", clientes.to_csv(index=False).encode("utf-8"), "clientes_segmentados.csv", "text/csv")
st.info("Los segmentos resumen compras de 2010–2011. No indican la situación actual de esas personas ni garantizan resultados comerciales futuros.")
