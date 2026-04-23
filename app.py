import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Detección de PPE",
    page_icon="🦺",
    layout="wide",
)

st.title("🦺 Detección de Equipos de Protección Personal (PPE)")
st.markdown("Sube una imagen o usa la cámara web para detectar objetos con tu modelo YOLOv8 entrenado.")

# ─────────────────────────────────────────────
# Sidebar – configuración del modelo
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    model_path = st.text_input(
        "Ruta del modelo (.pt)",
        value="best.pt",
        help="Ruta al archivo best.pt generado por el entrenamiento. "
             "Ejemplo: runs/ppe_model_pro/weights/best.pt"
    )

    confidence = st.slider(
        "Confianza mínima",
        min_value=0.1,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Predicciones con confianza menor a este valor serán ignoradas."
    )

    iou_threshold = st.slider(
        "Umbral IoU (NMS)",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Umbral para Non-Maximum Suppression."
    )

    show_labels = st.checkbox("Mostrar etiquetas", value=True)
    show_conf   = st.checkbox("Mostrar confianza", value=True)

    st.markdown("---")
    st.markdown("**Colores por clase** se asignan automáticamente.")

# ─────────────────────────────────────────────
# Cargar modelo (con caché)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_model(model_path)

if model is None:
    st.error(
        f"❌ No se encontró el modelo en **{model_path}**.\n\n"
        "Asegúrate de que el archivo `best.pt` esté en la misma carpeta que `app.py`, "
        "o ingresa la ruta correcta en la barra lateral.\n\n"
        "Ruta típica después del entrenamiento:  \n"
        "`/content/drive/MyDrive/ppe/runs/ppe_model_pro/weights/best.pt`"
    )
    st.stop()

st.success(f"✅ Modelo cargado: `{model_path}`")

# Paleta de colores para las clases
COLORS = [
    (255,  56,  56), (255, 157,  56), (255, 212,  56), (56, 255,  56),
    ( 56, 255, 157), ( 56, 212, 255), ( 56, 157, 255), (156,  56, 255),
    (255,  56, 212), (255, 255,  56),
]

def get_color(class_id: int):
    return COLORS[class_id % len(COLORS)]

# ─────────────────────────────────────────────
# Función de inferencia y dibujo
# ─────────────────────────────────────────────
def run_inference(image_bgr: np.ndarray):
    results = model.predict(
        source=image_bgr,
        conf=confidence,
        iou=iou_threshold,
        verbose=False,
    )[0]

    annotated = image_bgr.copy()
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id  = int(box.cls[0])
        conf_val = float(box.conf[0])
        label   = model.names[cls_id]
        color   = get_color(cls_id)

        # Cuadro
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Etiqueta
        if show_labels or show_conf:
            text_parts = []
            if show_labels:
                text_parts.append(label)
            if show_conf:
                text_parts.append(f"{conf_val:.2f}")
            text = " ".join(text_parts)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        detections.append({"Clase": label, "Confianza": f"{conf_val:.2%}",
                            "X1": x1, "Y1": y1, "X2": x2, "Y2": y2})

    return annotated, detections

# ─────────────────────────────────────────────
# Tabs: Imagen | Cámara
# ─────────────────────────────────────────────
tab_img, tab_cam = st.tabs(["🖼️ Imagen", "📷 Cámara en vivo"])

# ══════════════════ TAB IMAGEN ══════════════════
with tab_img:
    uploaded_file = st.file_uploader(
        "Sube una imagen (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_bgr   = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_image, use_column_width=True)

        with st.spinner("Procesando…"):
            annotated_bgr, detections = run_inference(img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader(f"Resultado — {len(detections)} detección(es)")
            st.image(annotated_rgb, use_column_width=True)

        if detections:
            st.markdown("### 📋 Detalle de detecciones")
            st.dataframe(detections, use_container_width=True)
        else:
            st.info("No se detectó ningún objeto con la confianza configurada.")

        # Botón de descarga
        _, img_encoded = cv2.imencode(".jpg", annotated_bgr)
        st.download_button(
            "⬇️ Descargar imagen anotada",
            data=img_encoded.tobytes(),
            file_name="prediccion.jpg",
            mime="image/jpeg",
        )



# ══════════════════ TAB CÁMARA ══════════════════
with tab_cam:
    st.info(
        "📷 Activa la cámara web para detección en tiempo real.\n\n"
        "Nota: En Streamlit Cloud es posible que debas permitir el acceso a la cámara en el navegador."
    )

    run_camera = st.checkbox("🟢 Activar cámara")

    if run_camera:
        frame_window = st.image([])
        info_area    = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No se pudo acceder a la cámara. Verifica los permisos.")
        else:
            stop_btn = st.button("🔴 Detener cámara")
            while not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning("No se pudo leer el frame.")
                    break
                annotated, detections = run_inference(frame)
                frame_window.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_column_width=True,
                )
                counts = {}
                for d in detections:
                    counts[d["Clase"]] = counts.get(d["Clase"], 0) + 1
                info_area.markdown(
                    "**Detecciones actuales:** " +
                    " | ".join([f"`{k}`: {v}" for k, v in counts.items()])
                    if counts else "**Sin detecciones**"
                )
            cap.release()