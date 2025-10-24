import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# --------------------------------------------------------------------------------------------------------------
# --- Model Loading and Constants ---

# You should adjust the model path if necessary
model_path = 'ev_cnn_mobile.keras'
try:
    # Use st.cache_resource to load the model only once
    @st.cache_resource
    def load_model(path):
        # Load the model with the custom object for MeanSquaredError
        return tf.keras.models.load_model(
            path, 
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
    model = load_model(model_path)
except Exception as e:
    # Use st.warning instead of st.error to allow the app to run (for the 'Knowledge' page)
    # but inform the user about the model status.
    st.warning(f"⚠️ Warning: Error loading model from '{model_path}'. The Detection page may not function.")
    model = None # Set model to None if loading fails

class_label = ["Artifact", "Ev eggs"]

# --------------------------------------------------------------------------------------------------------------
# --- Utility Functions (Same as before) ---

def drawbox(img, label, a, b, c, d, color):
    # a, b, c, d are (top, bottom, left, right)
    image = cv2.rectangle(img, (c, a), (d, b), color, 3)
    # Put text 10 pixels above the top-left corner
    image = cv2.putText(image, label, (c, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)
    return image

def compute_iou(box1, box2):
    # box1 and box2 are [y1, y2, x1, x2]
    y1 = max(box1[0], box2[0])
    y2 = min(box1[1], box2[1])
    x1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
    box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def nms(detections, iou_threshold):
    nms_dets = []
    for class_idx in set([d['class_idx'] for d in detections]):
        class_dets = [d for d in detections if d['class_idx'] == class_idx]
        class_dets = sorted(class_dets, key=lambda x: x['score'], reverse=True)
        keep = []
        while class_dets:
            curr = class_dets.pop(0) 
            keep.append(curr)
            class_dets = [
                d for d in class_dets
                if compute_iou(curr['bbox'], d['bbox']) < iou_threshold
            ]
        nms_dets.extend(keep)
    return nms_dets

def merge_connected_boxes_by_class(detections, merge_iou_threshold):
    merged = []
    for class_idx in set([d['class_idx'] for d in detections]):
        class_dets = [d for d in detections if d['class_idx'] == class_idx]
        used = set()
        groups = []
        for i, det in enumerate(class_dets):
            if i in used:
                continue
            group = [det]
            used.add(i)
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(class_dets):
                    if j in used:
                        continue
                    if any(compute_iou(d['bbox'], other['bbox']) > merge_iou_threshold for d in group):
                        group.append(other)
                        used.add(j)
                        changed = True
            groups.append(group)
        
        for group in groups:
            tops = [d['bbox'][0] for d in group]
            bottoms = [d['bbox'][1] for d in group]
            lefts = [d['bbox'][2] for d in group]
            rights = [d['bbox'][3] for d in group]
            merged_box = [min(tops), max(bottoms), min(lefts), max(rights)]
            max_score = max(d['score'] for d in group)
            merged.append({"bbox": merged_box, "class_idx": class_idx, "score": max_score})
    return merged

def ObjectDet(img, threshold, nms_threshold, merge_iou_threshold, model_instance):
    if model_instance is None:
        st.error("Model is not loaded. Cannot perform detection.")
        return []
        
    box_size_y, box_size_x, step_size = 500, 500, 50
    resize_input_y, resize_input_x = 64, 64
    img_h, img_w = img.shape[:2]

    coords = []
    patches = []
    # Sliding window to extract image patches
    for i in range(0, img_h - box_size_y + 1, step_size):
        for j in range(0, img_w - box_size_x + 1, step_size):
            img_patch = img[i:i+box_size_y, j:j+box_size_x]
            
            # Skip dark patches
            brightness = np.mean(cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY))
            if brightness < 50:
                continue
            
            # Prepare patch for model input
            resized_patch = cv2.resize(img_patch, (resize_input_x, resize_input_y))
            patches.append(resized_patch)
            coords.append((i, i + box_size_y, j, j + box_size_x)) # (y1, y2, x1, x2)
    
    if not patches:
        return []

    # Convert to NumPy array and normalize
    X = np.array(patches, dtype=np.float32) / 255.0

    # Model prediction
    predictions = model_instance.predict(X, verbose=0)
    
    detections = []
    for (y1, y2, x1, x2), prediction in zip(coords, predictions):
        class_idx = np.argmax(prediction)
        score = prediction[class_idx]

        if score >= threshold:
            bbox = [y1, y2, x1, x2]
            detections.append({
                "bbox": bbox,
                "class_idx": class_idx,
                "score": score
            })

    # Apply NMS and Merge
    nms_detections = nms(detections, nms_threshold)
    final_detections = merge_connected_boxes_by_class(nms_detections, merge_iou_threshold)
    
    return final_detections

# --------------------------------------------------------------------------------------------------------------
# --- Page Definitions ---

def knowledge_page():
    st.title("📚 ข้อมูลความรู้เกี่ยวกับพยาธิเข็มหมุด (Pinworm)")
    st.markdown("---")

    st.header("🦠 พยาธิเข็มหมุด (_Enterobius vermicularis_)")
    st.markdown("""
    พยาธิเข็มหมุด หรือเรียกอีกชื่อว่า พยาธิเส้นด้าย เป็นปรสิตที่พบบ่อยในลำไส้ใหญ่ของมนุษย์
    โดยเฉพาะใน **เด็กวัยเรียนและวัยก่อนเรียน** เป็นสาเหตุหลักของการติดเชื้อพยาธิในกลุ่มประเทศพัฒนาแล้ว
    """)
    
    st.subheader("วงจรชีวิตและการติดต่อ")
    st.markdown("""
    * **การติดเชื้อ:** เกิดจากการกลืนกินไข่พยาธิที่มองไม่เห็นด้วยตาเปล่าเข้าไป
    * **การแพร่เชื้อ:** ตัวเมียจะเคลื่อนที่มาวางไข่รอบ ๆ ทวารหนักในเวลากลางคืน ซึ่งทำให้เกิด **อาการคัน**
    * **การปนเปื้อน:** เมื่อผู้ป่วยเกา จะทำให้ไข่พยาธิติดที่มือ เล็บ และปนเปื้อนไปยังวัตถุอื่น ๆ เช่น ผ้าปูที่นอน ของเล่น อาหาร
    * **ติดต่อซ้ำ:** ไข่สามารถอยู่รอดภายนอกร่างกายได้ 2-3 สัปดาห์ และติดต่อกลับเข้าสู่ตัวเดิมหรือผู้อื่นได้ง่าย (Self-reinfection/External autoinfection)
    """)
    
    st.subheader("อาการที่พบ")
    st.markdown("""
    * **อาการคันอย่างรุนแรง** รอบทวารหนัก โดยเฉพาะในเวลากลางคืน
    * นอนไม่หลับ หรือหงุดหงิดจากการเกา
    * อาจมีอาการปวดท้อง, คลื่นไส้ หรือเบื่ออาหาร
    * ในเพศหญิง อาจพบการติดเชื้อในช่องคลอดหรือทางเดินปัสสาวะได้
    """)
    
    st.subheader("การวินิจฉัยและการรักษา")
    st.markdown("""
    * **การวินิจฉัย:** ใช้วิธี **Tape Test** (กดเทปใสรอบทวารหนักในตอนเช้าก่อนอาบน้ำ เพื่อเก็บไข่ไปตรวจด้วยกล้องจุลทรรศน์)
    * **การรักษา:** ใช้ยาถ่ายพยาธิ เช่น **Mebendazole** หรือ **Albendazole** โดยต้องรักษาซ้ำหลังจาก 2 สัปดาห์ เพื่อฆ่าตัวอ่อนที่ฟักออกมาใหม่ และควรรักษาคนในครอบครัวพร้อมกัน
    """)
    

def detection_page(model_instance):
    st.title("🔬 Pinworm Disease Diagnosis: Egg Detection")
    st.markdown("Upload a microscope image to detect **_Enterobius vermicularis_ (Ev) eggs**.")
    st.markdown("---")

    if model_instance is None:
        st.error("🚨 ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์ 'ev_cnn_mobile.keras' และลองใหม่อีกครั้ง")
        return

    # --- Detection Settings Sidebar (Moved from main) ---
    st.sidebar.header("⚙️ Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold (Min Score)", 0.0, 1.0, 0.8, 0.05)
    nms_iou_threshold = st.sidebar.slider("NMS IOU Threshold", 0.0, 1.0, 0.1, 0.05)
    merge_iou_threshold = st.sidebar.slider("Merge IOU Threshold", 0.0, 1.0, 0.05, 0.05)

    uploaded_file = st.file_uploader("Upload a Microscopic Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.subheader("Original Image")
        st.image(img_cv, channels="BGR", use_column_width=True)

        st.markdown("---")
        st.subheader("Detection Results")
        
        with st.spinner("Analyzing image for pinworm eggs..."):
            # Run detection
            results = ObjectDet(
                img_cv.copy(), 
                confidence_threshold, 
                nms_iou_threshold, 
                merge_iou_threshold,
                model_instance # Pass the loaded model instance
            )

        result_img = img_cv.copy()
        egg_count = 0
        
        if not results:
            st.info("No objects detected with the current settings. Try adjusting the Confidence Threshold in the sidebar.")
        else:
            for det in results:
                bbox = det['bbox']
                class_idx = det['class_idx']
                score = det['score']
                
                label = class_label[class_idx]
                color = (0, 255, 0) if label == "Ev eggs" else (0, 0, 255) # Green for eggs, Red for artifacts

                result_img = drawbox(result_img, f"{label}: {score:.2f}", bbox[0], bbox[1], bbox[2], bbox[3], color)
                
                if label == "Ev eggs":
                    egg_count += 1
            
            st.success(f"🎉 **Analysis Complete!** Found **{egg_count}** {class_label[1]}(s)!")

            st.image(result_img, channels="BGR", use_column_width=True)

# --------------------------------------------------------------------------------------------------------------
# --- Main App Logic ---

# Set the web application title
st.set_page_config(
    page_title="Pinworm Disease Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Menu Selection ---
st.sidebar.title("เมนูหลัก")
page_selection = st.sidebar.radio(
    "เลือกหน้าเว็บ:",
    ("1. ข้อมูลพยาธิเข็มหมุด (Knowledge)", "2. ตรวจจับไข่พยาธิ (Detection Model)")
)
st.sidebar.markdown("---")
st.sidebar.caption("By Pinworm Diagnosis Team")


# --- Page Routing ---
if page_selection == "1. ข้อมูลพยาธิเข็มหมุด (Knowledge)":
    knowledge_page()
elif page_selection == "2. ตรวจจับไข่พยาธิ (Detection Model)":
    detection_page(model)
