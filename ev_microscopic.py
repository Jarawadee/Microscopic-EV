import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import io

# --- 1. Streamlit Configuration ---
st.set_page_config(
    page_title="Pinworm Disease Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Constants (ค่าเฉพาะที่กำหนดไว้) ---
DEFAULT_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_MERGE_IOU_THRESHOLD = 0.2

st.title("🔬 Pinworm Disease Diagnosis App")
st.header("ยินดีต้อนรับ!")
st.markdown("""
แอปพลิเคชันนี้ใช้ AI ในการตรวจหาไข่พยาธิเข็มหมุดจากภาพถ่ายกล้องจุลทรรศน์
""")

add_selectbox = st.sidebar.selectbox(
    "เลือกเมนูการใช้งาน:",
    ("หน้าหลัก/ความรู้เกี่ยวกับพยาธิเข็มหมุด", "🔎 AI detection")
)

# --- 3. Model Loading ---
@st.cache_resource()
def load_model():
    model_path = 'ev_cnn_mobile.keras'
    try:
        # โหลดโมเดลพร้อมจัดการ custom_objects
        model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดไฟล์โมเดลได้: {e}")
        return None

model = load_model()
class_label = ["Artifact", "Ev eggs"]

# --- 4. Utility Functions ---

def compute_iou(box1, box2):
    y1, y2, x1, x2 = max(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
    box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def nms(detections, iou_threshold):
    nms_dets = []
    class_indices = set([d['class_idx'] for d in detections])
    for class_idx in class_indices:
        class_dets = sorted([d for d in detections if d['class_idx'] == class_idx], key=lambda x: x['score'], reverse=True)
        while class_dets:
            curr = class_dets.pop(0)
            nms_dets.append(curr)
            class_dets = [d for d in class_dets if compute_iou(curr['bbox'], d['bbox']) < iou_threshold]
    return nms_dets

def merge_connected_boxes_by_class(detections, merge_iou_threshold):
    merged = []
    class_indices = set([d['class_idx'] for d in detections])
    for class_idx in class_indices:
        class_dets = [d for d in detections if d['class_idx'] == class_idx]
        used = set()
        for i, det in enumerate(class_dets):
            if i in used: continue
            group = [det]
            used.add(i)
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(class_dets):
                    if j not in used and any(compute_iou(d['bbox'], other['bbox']) > merge_iou_threshold for d in group):
                        group.append(other)
                        used.add(j)
                        changed = True
            
            tops, bottoms, lefts, rights = [d['bbox'][0] for d in group], [d['bbox'][1] for d in group], [d['bbox'][2] for d in group], [d['bbox'][3] for d in group]
            merged.append({"bbox": [min(tops), max(bottoms), min(lefts), max(rights)], "class_idx": class_idx, "score": max(d['score'] for d in group)})
    return merged

def ObjectDet(img, threshold, nms_threshold, merge_iou_threshold):
    if model is None: return img
    
    box_size, step_size = 500, 50
    img_h, img_w = img.shape[:2]
    coords, patches = [], []

    for i in range(0, img_h - box_size + 1, step_size):
        for j in range(0, img_w - box_size + 1, step_size):
            patch = img[i:i+box_size, j:j+box_size]
            if np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)) < 50: continue
            patches.append(cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA))
            coords.append((i, j))

    if not patches: return img

    y_out = model.predict(np.array(patches), batch_size=64, verbose=0)
    detections = []
    for idx, pred in enumerate(y_out):
        for class_idx in range(1, len(class_label)): # ข้าม Artifact (0)
            if pred[class_idx] > threshold:
                a, c = coords[idx]
                detections.append({"bbox": [a, a + box_size, c, c + box_size], "score": float(pred[class_idx]), "class_idx": class_idx})

    results = nms(detections, nms_threshold)
    if merge_iou_threshold > 0:
        results = merge_connected_boxes_by_class(results, merge_iou_threshold)

    for det in results:
        a, b, c, d = det['bbox']
        cv2.rectangle(img, (c, a), (d, b), (0, 255, 0), 3)
        cv2.putText(img, f"{class_label[det['class_idx']]}: {det['score']:.2f}", (c, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
    return img

# --- 5. UI Flow ---

if add_selectbox == "หน้าหลัก/ความรู้เกี่ยวกับพยาธิเข็มหมุด":
    st.markdown("### 📚 ลักษณะของไข่พยาธิเข็มหมุด")
    st.write("ไข่ของพยาธิเข็มหมุดจะมีลักษณะรี ผิวเรียบ และมีด้านหนึ่งที่แบนกว่าอีกด้าน (คล้ายตัว D)")
    # 

[Image of Enterobius vermicularis egg morphology]

    st.info("ไข่พยาธิเข็มหมุดมักตรวจพบได้จากการทำ Tape Test บริเวณรอบทวารหนัก")

elif add_selectbox == "🔎 AI detection":
    uploaded_file = st.file_uploader("อัปโหลดภาพ (PNG, JPG, JPEG, TIF)", type=["png", "jpg", "jpeg", "tif"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ภาพต้นฉบับ")
                st.image(image, use_container_width=True)

            if model:
                with st.spinner('AI กำลังวิเคราะห์...'):
                    # เรียกใช้ ObjectDet พร้อมค่าคงที่ที่กำหนดไว้ด้านบน
                    res_bgr = ObjectDet(image_bgr, DEFAULT_THRESHOLD, DEFAULT_NMS_THRESHOLD, DEFAULT_MERGE_IOU_THRESHOLD)
                    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("ผลการวิเคราะห์")
                    st.image(res_rgb, use_container_width=True)
            else:
                st.warning("โมเดลไม่พร้อมใช้งาน")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
