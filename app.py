import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Vision · Detección de EPP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow:wght@300;400;600;700;800&family=Barlow+Condensed:wght@700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0f14;
    color: #e8eaf0;
    font-family: 'Barlow', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111318 !important;
    border-right: 1px solid #1e2130;
}
.ppe-header { padding: 2.5rem 0 1.5rem 0; border-bottom: 1px solid #1e2130; margin-bottom: 2rem; }
.ppe-header .badge {
    display: inline-block; background: #00e5a0; color: #0d0f14;
    font-family: 'Space Mono', monospace; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.15em; padding: 3px 10px; border-radius: 2px;
    margin-bottom: 0.8rem; text-transform: uppercase;
}
.ppe-header h1 {
    font-family: 'Barlow Condensed', sans-serif; font-size: 3.2rem;
    font-weight: 800; letter-spacing: -0.02em; color: #ffffff;
    margin: 0 0 0.4rem 0; line-height: 1;
}
.ppe-header h1 span { color: #00e5a0; }
.ppe-header p { color: #6b7280; font-size: 1rem; font-weight: 300; margin: 0; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Barlow Condensed', sans-serif; color: #ffffff;
    letter-spacing: 0.05em; text-transform: uppercase; font-size: 1rem;
}
[data-testid="stSidebar"] label {
    color: #9ca3af !important; font-size: 0.82rem !important;
    font-family: 'Space Mono', monospace !important; letter-spacing: 0.05em;
}

[data-testid="stTabs"] button {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: #6b7280 !important; padding: 0.6rem 1.4rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5a0 !important; border-bottom: 2px solid #00e5a0 !important;
}

.result-card {
    background: #111318; border: 1px solid #1e2130; border-radius: 8px;
    padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.result-card h4 {
    font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: #6b7280;
    margin: 0 0 0.3rem 0;
}
.result-card .value {
    font-family: 'Space Mono', monospace; font-size: 2rem;
    font-weight: 700; color: #00e5a0; line-height: 1;
}

.img-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #6b7280; margin-bottom: 0.5rem;
}

.info-box {
    background: #111318; border: 1px solid #1e2130;
    border-left: 3px solid #00e5a0; border-radius: 4px;
    padding: 1rem 1.2rem; font-size: 0.88rem;
    color: #9ca3af; margin-bottom: 1.2rem;
}

.stDownloadButton > button {
    background: transparent !important; border: 1px solid #00e5a0 !important;
    color: #00e5a0 !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important; letter-spacing: 0.08em !important;
    border-radius: 4px !important; padding: 0.5rem 1.2rem !important;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    background: #00e5a0 !important; color: #0d0f14 !important;
}

[data-testid="stAlert"] {
    background: #111318 !important; border: 1px solid #1e2130 !important;
    border-radius: 6px !important; color: #9ca3af !important;
}
[data-testid="stSuccess"] {
    background: #0a1f17 !important; border: 1px solid #00e5a0 !important;
    border-radius: 6px !important; color: #00e5a0 !important;
}
[data-testid="stError"] {
    background: #1f0a0a !important; border: 1px solid #ef4444 !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"] {
    background: #111318 !important; border: 1px dashed #1e2130 !important;
    border-radius: 8px !important;
}
hr { border-color: #1e2130 !important; }
.version-tag {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    color: #374151; text-align: center; padding-top: 1rem; letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="ppe-header">
    <div class="badge">🛡️ Sistema de Visión Industrial</div>
    <h1>PPE <span>VISION</span></h1>
    <p>Detección automática de Equipos de Protección Personal mediante YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parámetros del Modelo")
    st.markdown("---")

    model_path = st.text_input(
        "Archivo del modelo (.pt)",
        value="best.pt",
        help="Ruta al archivo best.pt. Por defecto busca en la misma carpeta que app.py.",
    )

    st.markdown("---")
    st.markdown("### 🎯 Umbrales de Detección")

    confidence = st.slider(
        "Confianza mínima",
        min_value=0.05, max_value=1.0, value=0.40, step=0.05,
        help="Detecciones con puntuación menor a este valor serán descartadas.",
    )
    iou_threshold = st.slider(
        "Umbral IoU (NMS)",
        min_value=0.1, max_value=1.0, value=0.45, step=0.05,
        help="Controla la supresión de cuadros solapados.",
    )

    st.markdown("---")
    st.markdown("### 🖍️ Visualización")
    show_labels = st.checkbox("Mostrar nombre de clase", value=True)
    show_conf   = st.checkbox("Mostrar puntuación de confianza", value=True)

    st.markdown("---")
    st.markdown('<div class="version-tag">PPE VISION · YOLOv8 · v1.0</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CARGAR MODELO
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_model(model_path)

if model is None:
    st.error(
        f"**Modelo no encontrado:** `{model_path}`\n\n"
        "Asegúrate de que `best.pt` esté en la misma carpeta que `app.py`, "
        "o escribe la ruta correcta en la barra lateral."
    )
    st.stop()

st.success(f"✓ Modelo cargado correctamente — `{model_path}` · {len(model.names)} clases detectables")

# ─────────────────────────────────────────────────────────────
# PALETA Y UTILIDADES
# ─────────────────────────────────────────────────────────────
COLORS = [
    (0, 229, 160), (255, 87, 51), (51, 181, 255), (255, 214, 51),
    (180, 51, 255), (255, 51, 130), (51, 255, 214), (255, 140, 51),
    (51, 255, 87),  (255, 51, 51),
]

def get_color(class_id: int):
    return COLORS[class_id % len(COLORS)]

def run_inference(image_bgr: np.ndarray):
    results = model.predict(
        source=image_bgr, conf=confidence, iou=iou_threshold, verbose=False,
    )[0]

    annotated  = image_bgr.copy()
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id   = int(box.cls[0])
        conf_val = float(box.conf[0])
        label    = model.names[cls_id]
        color    = get_color(cls_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if show_labels or show_conf:
            parts = []
            if show_labels:  parts.append(label.upper())
            if show_conf:    parts.append(f"{conf_val:.0%}")
            text = "  ".join(parts)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (13, 15, 20), 2)

        detections.append({
            "Clase": label, "Confianza": f"{conf_val:.1%}",
            "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
            "Ancho (px)": x2 - x1, "Alto (px)": y2 - y1,
        })

    return annotated, detections


def show_detection_stats(detections):
    if not detections:
        st.markdown('<div class="info-box">⚠️ No se detectó ningún objeto con los umbrales configurados. Prueba reduciendo la confianza mínima en la barra lateral.</div>', unsafe_allow_html=True)
        return

    counts = {}
    for d in detections:
        counts[d["Clase"]] = counts.get(d["Clase"], 0) + 1

    best = max(detections, key=lambda d: float(d["Confianza"].replace("%", "")))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="result-card"><h4>Total detectado</h4><div class="value">{len(detections)}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="result-card"><h4>Clases distintas</h4><div class="value">{len(counts)}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="result-card"><h4>Mayor confianza</h4><div class="value">{best["Confianza"]}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Detalle de detecciones**")
    st.dataframe(detections, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_img, tab_video, tab_cam = st.tabs([
    "🖼️   Imagen",
    "🎬   Video",
    "📷   Cámara",
])

# ══════════════════ TAB: IMAGEN ══════════════════
with tab_img:
    st.markdown('<div class="info-box">Sube una imagen en JPG, PNG o WEBP. El modelo analizará y marcará los equipos de protección detectados.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Arrastra o selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        img_bgr   = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        with st.spinner("Analizando imagen con YOLOv8…"):
            annotated_bgr, detections = run_inference(img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<p class="img-label">// Imagen original</p>', unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True)
        with col2:
            st.markdown(f'<p class="img-label">// Resultado — {len(detections)} detección(es)</p>', unsafe_allow_html=True)
            st.image(annotated_rgb, use_column_width=True)

        st.markdown("---")
        show_detection_stats(detections)

        _, img_enc = cv2.imencode(".jpg", annotated_bgr)
        st.download_button("⬇ Descargar imagen anotada", data=img_enc.tobytes(),
                           file_name="ppe_resultado.jpg", mime="image/jpeg")

# ══════════════════ TAB: VIDEO ══════════════════
with tab_video:
    st.markdown('<div class="info-box">Sube un video MP4, AVI o MOV. El sistema procesará cada fotograma y generará un video anotado listo para descargar.</div>', unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Arrastra o selecciona un video",
        type=["mp4", "avi", "mov"],
        key="vid",
        label_visibility="collapsed",
    )

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        out_path = tmp_path.replace(".mp4", "_anotado.mp4")
        cap    = cv2.VideoCapture(tmp_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        barra     = st.progress(0, text="Iniciando procesamiento…")
        vista_frm = st.empty()
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            anotado, _ = run_inference(frame)
            writer.write(anotado)
            frame_num += 1
            pct = int(frame_num / max(total, 1) * 100)
            barra.progress(pct, text=f"Fotograma {frame_num} / {total} — {pct}%")
            if frame_num % 15 == 0:
                vista_frm.image(cv2.cvtColor(anotado, cv2.COLOR_BGR2RGB),
                                caption=f"Vista previa · fotograma {frame_num}",
                                use_column_width=True)

        cap.release()
        writer.release()
        barra.progress(100, text="✓ Procesamiento completado")

        with open(out_path, "rb") as f:
            st.download_button("⬇ Descargar video anotado", data=f.read(),
                               file_name="ppe_video_resultado.mp4", mime="video/mp4")

        os.unlink(tmp_path)
        os.unlink(out_path)

# ══════════════════ TAB: CÁMARA ══════════════════
with tab_cam:
    st.markdown('<div class="info-box">Captura una foto con tu cámara. El modelo analizará la imagen al instante y mostrará los equipos de protección detectados.</div>', unsafe_allow_html=True)

    foto = st.camera_input("Capturar foto", label_visibility="collapsed")

    if foto is not None:
        pil_image = Image.open(foto).convert("RGB")
        img_bgr   = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        with st.spinner("Procesando captura…"):
            annotated_bgr, detections = run_inference(img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.markdown(f'<p class="img-label">// Resultado — {len(detections)} detección(es)</p>', unsafe_allow_html=True)
        st.image(annotated_rgb, use_column_width=True)
        st.markdown("---")
        show_detection_stats(detections)

        _, img_enc = cv2.imencode(".jpg", annotated_bgr)
        st.download_button("⬇ Descargar imagen anotada", data=img_enc.tobytes(),
                           file_name="ppe_camara_resultado.jpg", mime="image/jpeg")