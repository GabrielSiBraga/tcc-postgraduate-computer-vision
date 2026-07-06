"""Interface Streamlit para demonstração do pipeline."""

from __future__ import annotations

import base64
import os

import httpx
import streamlit as st

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")


def _b64_image(data: str) -> bytes:
    return base64.b64decode(data)


st.set_page_config(page_title="Leitura de Hidrômetro", layout="wide")
st.title("Pipeline Híbrido - Detectron2 + Florence-2 QLoRA")
st.caption("Upload de foto bruta → detecção → crop completo → CLAHE → leitura estruturada")

uploaded = st.file_uploader("Envie uma foto do hidrômetro", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagem original")
        st.image(uploaded, use_container_width=True)

    if st.button("Executar pipeline", type="primary"):
        with st.spinner("Processando..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            try:
                response = httpx.post(f"{API_URL}/predict", files=files, timeout=120.0)
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                st.error(f"Erro na API: {exc}")
                st.stop()

        debug = payload.get("debug", {})
        leitura = payload.get("leitura", {})

        col2.subheader("Overlay (display + hidrômetro completo)")
        if debug.get("overlay_base64"):
            st.image(_b64_image(debug["overlay_base64"]), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Crop CLAHE")
            if debug.get("crop_base64"):
                st.image(_b64_image(debug["crop_base64"]), use_container_width=True)
        with c2:
            st.subheader("Resultado")
            st.metric("Inteiro (m³)", leitura.get("inteiro", "-"))
            st.metric("Decimal", leitura.get("decimal", "-"))
            st.metric("Completo (visor)", leitura.get("completo", "-"))
            st.metric("Fabricante", payload.get("fabricante", "-"))
            st.metric("Estado", payload.get("estado", "-"))
            st.metric("Latência (ms)", payload.get("latency_ms", "-"))

        with st.expander("Debug"):
            st.json(payload)
