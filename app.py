import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from collections import Counter
import supervision as sv
import pandas as pd
import io

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi & Penghitungan Kendaraan",
    page_icon="🚗",
    layout="wide"
)

# --- FUNGSI-FUNGSI HELPER  ---
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file '{model_path}' ada. Error: {e}")
        return None

def process_detection(model, image_pil, confidence):
    results = model.predict(image_pil, conf=confidence, verbose=False) 
    result = results[0] 
    detections = sv.Detections.from_ultralytics(result) 
    
    
    class_names = result.names
    
    if detections.class_id is not None:
        detections.data['class_name'] = [class_names[class_id] for class_id in detections.class_id]
    else:
        detections.data['class_name'] = [] 
    
    return detections

def analyze_counts(detections):
    class_counts = Counter(detections.data['class_name'])
    return class_counts

def annotate_image(image_pil, detections):
    image_np = np.array(image_pil)
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)
    
    annotated_image_np = box_annotator.annotate(scene=image_np.copy(), detections=detections)
    
    annotated_image_np = label_annotator.annotate(scene=annotated_image_np, detections=detections)
    
    return Image.fromarray(annotated_image_np)

def display_results(class_counts):
    if not class_counts:
        st.warning("Tidak ada kendaraan yang terdeteksi.")
        return

    total_vehicles = sum(class_counts.values())
    st.metric(label="Total Kendaraan Terdeteksi", value=total_vehicles)
    
    count_df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Jumlah'])
    count_df.index.name = "Kelas Kendaraan"
    st.dataframe(count_df, use_container_width=True)

# Judul Utama
st.title("Sistem Deteksi dan Penghitungan Kendaraan")
st.markdown("Unggah gambar jalan raya untuk mendeteksi bus, car, dan van.")

# Sidebar
with st.sidebar:
    st.header("Upload & Setting")
    
    uploaded_file = st.file_uploader(
        "1. Unggah file gambar...", 
        type=["jpg", "jpeg", "png"]
    )
    
    confidence_threshold = st.slider(
        "2. Atur Confidence", 
        0.0, 1.0, 0.25, 0.05,
        help="Model hanya akan menampilkan objek yang terdeteksi dengan confidence di atas nilai ini."
    )

# Muat model
MODEL_PATH = "model/best_model_intishar.pt" 
model = load_model(MODEL_PATH) 

if model and uploaded_file is not None:
    
    image_pil = Image.open(io.BytesIO(uploaded_file.getvalue()))
    
    with st.spinner("Model sedang mendeteksi..."):
        
        detections = process_detection(model, image_pil, confidence_threshold)
        annotated_image = annotate_image(image_pil, detections)
        class_counts = analyze_counts(detections)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, caption="Gambar Asli", use_container_width=True)
    with col2:
        st.image(annotated_image, caption="Gambar Hasil Deteksi", use_container_width=True)

    st.header("Hasil Analisis: Penghitungan Objek")
    display_results(class_counts)
        
elif not model:
    st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Harap unduh model 'best_model_intishar.pt' Anda dari GDrive dan letakkan di dalam folder 'model/'.")
else:
    st.info("Silakan unggah gambar di sidebar untuk memulai deteksi.")