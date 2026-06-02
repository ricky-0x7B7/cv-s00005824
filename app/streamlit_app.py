"""Demo Streamlit: carica un'immagine, esegue il rilevamento DPI + compliance.

Esecuzione con::

    streamlit run app/streamlit_app.py

Web app opzionale per crediti extra (WP8 / criterio "deployment come web app").
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import streamlit as st

from src.compliance import assess_compliance, summarize_compliance
from src.config import SH17_CLASSES
from src.inference import load_model, predict_image
from src.postprocess import filter_detections
from src.visualization import draw_compliance, draw_detections

st.set_page_config(page_title="Sicurezza DPI in cantiere", layout="wide")
st.title("🦺 Monitor di sicurezza DPI in cantiere")
st.caption("Rilevamento YOLO + motore di compliance basato su regole sul dataset SH17.")

with st.sidebar:
    st.header("Impostazioni")
    weights = st.text_input("Percorso dei pesi", value="models/best.pt")
    conf = st.slider("Soglia di confidenza", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("Soglia IoU per NMS", 0.1, 0.9, 0.5, 0.05)
    device = st.selectbox("Device", ["auto", "mps", "cuda", "cpu"], index=0)


@st.cache_resource(show_spinner=True)
def get_model(path: str):
    return load_model(path)


uploaded = st.file_uploader("Carica un'immagine di un cantiere", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if not Path(weights).exists():
        st.error(f"Pesi non trovati in '{weights}'. Addestra YOLO o scarica prima best.pt.")
        st.stop()

    model = get_model(weights)
    detections = predict_image(model, image, conf=conf, iou=iou, device=device)
    detections = filter_detections(detections, conf_threshold=conf)
    assessments = assess_compliance(detections)

    annotated = draw_detections(image, detections, SH17_CLASSES)
    annotated = draw_compliance(annotated, assessments)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Rilevamenti + compliance",
                 use_container_width=True)
    with col2:
        st.subheader("Lavoratori")
        st.json(summarize_compliance(assessments))
        st.subheader("Rilevamenti")
        rows = [{"class": SH17_CLASSES[d["cls"]], "conf": round(d["conf"], 3)} for d in detections]
        st.dataframe(rows, use_container_width=True)
else:
    st.info("Carica un'immagine per iniziare.")
