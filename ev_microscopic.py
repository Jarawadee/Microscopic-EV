import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- 1. Config หลักของแอป (ต้องอยู่บรรทัดแรกๆ) ---
st.set_page_config(
    page_title="Pinworm Diagnosis App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ส่วนที่ 1: เตรียมฟังก์ชันสำหรับ AI และ Model
# ==========================================

@st.cache_resource()
def load_model():
    model_path = 'ev_cnn_mobile.keras'
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

model = load_model()
class_label = ["Artifact", "Ev eggs"]

def drawbox(img, label, a, b, c, d, color):
    image = cv2.rectangle(img, (c, a), (d, b), color, 3)
    image = cv2.putText(image, label, (c, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 3)
    return image

def compute_iou(box1, box2):
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
    if union_area == 0: return 0
    return inter_area / union_area

def nms(detections, iou_threshold):
    nms_dets = []
    if not detections: return []
    class_indices = set([d['class_idx'] for d in detections])
    for class_idx in class_indices:
        class_dets = [d for d in detections if d['class_idx'] == class_idx]
        class_dets = sorted(class_dets, key=lambda x: x['score'], reverse=True)
        keep = []
        while class_dets:
            curr = class_dets.pop(0)
            keep.append(curr)
            class_dets = [d for d in class_dets if compute_iou(curr['bbox'], d['bbox']) < iou_threshold]
        nms_dets.extend(keep)
    return nms_dets

def merge_connected_boxes_by_class(detections, merge_iou_threshold):
    merged = []
    if not detections: return []
    class_indices = set([d['class_idx'] for d in detections])
    for class_idx in class_indices:
        class_dets = [d for d in detections if d['class_idx'] == class_idx]
        used = set()
        groups = []
        for i, det in enumerate(class_dets):
            if i in used: continue
            group = [det]
            used.add(i)
            changed = True
            while changed:
                changed = False
                newly_added = []
                for j, other in enumerate(class_dets):
                    if j in used: continue
                    if any(compute_iou(d['bbox'], other['bbox']) > merge_iou_threshold for d in group):
                        newly_added.append((j, other))
                if newly_added:
                    for j, other in newly_added:
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

def ObjectDet(img, threshold, nms_threshold, merge_iou_threshold):
    box_size_y, box_size_x, step_size = 500, 500, 50
    resize_input_y, resize_input_x = 64, 64
    img_h, img_w = img.shape[:2]
    coords = []
    patches = []
    
    for i in range(0, img_h - box_size_y + 1, step_size):
        for j in range(0, img_w - box_size_x + 1, step_size):
            img_patch = img[i:i+box_size_y, j:j+box_size_x]
            brightness = np.mean(cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY))
            if brightness < 50: continue
            img_patch_resized = cv2.resize(img_patch, (resize_input_y, resize_input_x), interpolation=cv2.INTER_AREA)
            patches.append(img_patch_resized)
            coords.append((i, j))

    if not patches: return img
    patches = np.array(patches)
    if model is None: return img
        
    y_out = model.predict(patches, batch_size=64, verbose=0)
    detections = []
    for idx, pred in enumerate(y_out):
        for class_idx in range(len(class_label)):
            score = pred[class_idx]
            if score > threshold and class_idx != 0:
                a, c = coords[idx]
                b, d = a + box_size_y, c + box_size_x
                detections.append({"bbox": [a, b, c, d], "score": float(score), "class_idx": class_idx})

    nms_detections = nms(detections, iou_threshold=nms_threshold)
    if merge_iou_threshold is not None and merge_iou_threshold > 0:
        final_detections = merge_connected_boxes_by_class(nms_detections, merge_iou_threshold=merge_iou_threshold)
    else:
        final_detections = nms_detections

    img_output = img.copy()
    colors = [(0,255,0), (255,0,0)] 
    for det in final_detections:
        a, b, c, d = det['bbox']
        class_idx = det['class_idx']
        label = f"{class_label[class_idx]}: {det['score']:.2f}"
        color = colors[class_idx % len(colors)]
        img_output = drawbox(img_output, label, a, b, c, d, color)
    return img_output

# ==========================================
# ส่วนที่ 2: สร้างฟังก์ชันสำหรับแต่ละหน้า (Page Functions)
# ==========================================

# 1. ฟังก์ชันหน้า Homepage
# แก้ไขฟังก์ชัน page_home

def page_home():
    # จัด Layout ให้อยู่กึ่งกลาง (เว้นซ้ายขวาอย่างละ 1 ส่วน เนื้อหาตรงกลาง 2 ส่วน)
    col1, main_col, col3 = st.columns([1, 2, 1])

    with main_col:
        # --- ส่วนหัวและรูปภาพ ---
        st.markdown("<h1 style='text-align: center;'>🔬 Pinworm Diagnosis App</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>ระบบช่วยคัดกรองและให้ความรู้โรคพยาธิเข็มหมุดด้วย AI</p>", unsafe_allow_html=True)
        
        st.image("Gemini_Generated_Image_i4nkkdi4nkkdi4nk.png", use_column_width=True)
        
        st.divider() # เส้นขีดคั่น

        # --- ส่วนที่ 2: How it works (ขั้นตอนง่ายๆ) ---
        st.subheader("💡 ขั้นตอนการใช้งาน")
        step1, step2, step3 = st.columns(3)
        with step1:
            st.markdown("**1. เตรียมภาพ**")
            st.caption("ถ่ายภาพจากกล้องจุลทรรศน์ (Tape Test)")
        with step2:
            st.markdown("**2. อัปโหลด**")
            st.caption("ไปที่เมนู AI Detection และเลือกไฟล์รูป")
        with step3:
            st.markdown("**3. ดูผลลัพธ์**")
            st.caption("AI จะระบุตำแหน่งไข่พยาธิให้ทันที")

        st.divider()

        # --- ส่วนที่ 3: ปุ่มทางลัด (Call to Action) ---
        # หมายเหตุ: ปุ่ม switch_page ต้องใช้คู่กับ st.navigation หรือ setup ที่ถูกต้อง
        # ถ้ายังทำปุ่มลิ้งค์ไม่ได้ ให้ใช้ข้อความเชิญชวนแทน
        st.info("👉 พร้อมแล้วใช่ไหม? ไปที่เมนู **'AI Detection'** ทางด้านซ้ายเพื่อเริ่มวิเคราะห์")

        # --- ส่วนที่ 1: Medical Disclaimer (สำคัญมาก!) ---
        st.warning("""
        **⚠️ คำเตือน:** ผลลัพธ์จาก AI นี้ใช้เพื่อการ **คัดกรองเบื้องต้นเท่านั้น** ไม่สามารถใช้แทนการวินิจฉัยโดยแพทย์หรือนักเทคนิคการแพทย์ได้ 
        หากมีอาการผิดปกติ ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อการรักษาที่ถูกต้อง
        """)


# 2. ฟังก์ชันเนื้อหาความรู้ย่อย
def content_general_info():
    st.header("📄 ข้อมูลทั่วไป")
    st.markdown("""
    **พยาธิเข็มหมุด (_Enterobius vermicularis_)** เป็นพยาธิตัวกลมขนาดเล็ก สีขาว คล้ายเส้นด้าย พบบ่อยในเด็กทั่วโลก
    พยาธิตัวเมียจะอาศัยอยู่ในลำไส้ใหญ่ และจะคลานออกมาวางไข่รอบๆ ทวารหนักในเวลากลางคืน ทำให้เกิดอาการคัน
    """)
    st.subheader("Life cycle")
    st.image("https://www.cdc.gov/dpdx/enterobiasis/modules/Enterobius_LifeCycl_lg.jpg" )
    st.markdown("""หนอนพยาธิเข็มหมุดตัวเมียที่โตเต็มวัยจะออกมาวางไข่บนรอยพับผิวหนังรอบทวารหนัก ทำให้เกิดอาการคันบริเวณทวารหนักเวลากลางคืน
    ซึ่งการติดเชื้อเกิดจากการปนเปื้อนตนเอง เช่น การกลืนกินไข่เข้าปากผ่านมือที่เกาบริเวณรอบทวารหนักแล้วไปหยิบจับอาหาร) หรือผ่านการสัมผัสกับไข่ในสิ่งแวดล้อม 
    (เช่น พื้นผิวที่ปนเปื้อน เสื้อผ้า ผ้าปูที่นอน ฯลฯ) 
    หลังจากกลืนไข่ที่ติดเชื้อแล้ว ตัวอ่อนจะฟักออกมาอาศัยในลำไส้เล็ก และตัวเต็มวัยจะอยู่ในลำไส้ใหญ่ส่วนต้น 
  และสามารถติดเชื้อย้อนกลับ 
โดยตัวอ่อนที่ฟักใหม่จากผิวหนังบริเวณทวารหนักอาจจะกลับเข้าไปในทวารหนักได้
""")

def content_symptoms():
    st.header("🤒 อาการของโรค")
    st.markdown("""
    * **อาการคัน:** บริเวณทวารหนัก โดยเฉพาะตอนกลางคืน
    * **นอนหลับไม่สนิท:** เด็กอาจงอแง พลิกตัวไปมา หรือนอนกัดฟัน
    * **อาการทางเดินอาหาร:** ปวดท้องเป็นครั้งคราว คลื่นไส้ เบื่ออาหาร
    """)

def content_prevention():
    st.header("🛡️ การป้องกันและการรักษา")
    st.info("สุขอนามัยที่ดีคือหัวใจสำคัญของการป้องกัน")
    st.markdown("""
    1. **ล้างมือ:** ล้างมือให้สะอาดด้วยสบู่ก่อนทานอาหารและหลังเข้าห้องน้ำ
    2. **ตัดเล็บ:** ตัดเล็บให้สั้นเพื่อลดการสะสมของไข่พยาธิตามซอกเล็บ
    3. **ทำความสะอาด:** ซักผ้าปูที่นอน เสื้อผ้า ชุดนอน ด้วยน้ำร้อนเพื่อฆ่าไข่พยาธิ
    4. **ทานยา:** หากพบสมาชิกในบ้านติดเชื้อ ควรทานยาถ่ายพยาธิพร้อมกันทุกคน
    """)

# ==========================================
# ส่วนที่ 3: สร้าง Wrapper Functions สำหรับหน้าต่างๆ
# ==========================================

# 🟢 ฟังก์ชันหน้า Knowledge (รวม Tab)
def page_knowledge_hub():
    st.title("📚 ความรู้เกี่ยวกับพยาธิเข็มหมุด")
    
    # สร้าง Tab 3 อัน
    tab1, tab2, tab3 = st.tabs(["ข้อมูลทั่วไป", "อาการ", "การป้องกัน"])
    
    with tab1:
        content_general_info()
    with tab2:
        content_symptoms()
    with tab3:
        content_prevention()

# 🟢 ฟังก์ชันหน้า AI Detection
def page_ai_detect():
    st.title("🔎 AI Detection")
    st.markdown("โปรดอัปโหลดภาพจากกล้องจุลทรรศน์ (Tape Test) เพื่อทำการวิเคราะห์")

    # Fixed Parameters
    detection_threshold = 0.95
    nms_threshold = 0.3
    merge_iou_threshold = 0.5
    st.info(f"⚙️ **System Parameters:** Confidence > {detection_threshold}, NMS = {nms_threshold}, Merge = {merge_iou_threshold}")

    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (PNG, JPG, JPEG, TIF)", type=["png", "jpg", "jpeg", "tif"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ภาพต้นฉบับ")
                st.image(image_np, caption=uploaded_file.name, use_column_width=True)

            if model is not None:
                with st.spinner('กำลังวิเคราะห์ภาพด้วย AI...'):
                    output_img_bgr = ObjectDet(image_bgr, detection_threshold, nms_threshold, merge_iou_threshold)
                
                output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
                with col2:
                    st.subheader("ผลการวิเคราะห์")
                    st.image(output_img_rgb, caption="AI Result", use_column_width=True)
            else:
                st.warning("Model not loaded (ev_cnn_mobile.keras not found).")
        except Exception as e:
            st.error(f"Error: {e}")

# 🟢 ฟังก์ชันหน้า Dataset
def page_dataset():
    st.header("📊 Dataset Information")
    st.write("ข้อมูลเกี่ยวกับ Dataset ที่ใช้เทรนโมเดลจะแสดงที่นี่...")

# ==========================================
# ส่วนที่ 4: ระบบนำทาง (Navigation)
# ==========================================

# 1. กำหนดหน้า (Pages)
# - หน้าแรก
p_home = st.Page(page_home, title="หน้าหลัก", icon="🏠")

# - หน้าความรู้
p_knowledge = st.Page(page_knowledge_hub, title="ความรู้เกี่ยวกับโรค", icon="📚")

# - หน้าเครื่องมือ
p_tool_ai = st.Page(page_ai_detect, title="AI Detection", icon="🔎")
p_tool_data = st.Page(page_dataset, title="Dataset", icon="📊")
                      
# 2. จัดกลุ่มหน้าลงในเมนู Sidebar
pg = st.navigation({
    "หน้าหลัก": [p_home],
    "ความรู้เกี่ยวกับโรคพยาธิเข็มหมุด": [p_knowledge],
    "เครื่องมือและข้อมูล": [p_tool_ai, p_tool_data]
})

# 3. รันระบบนำทาง
pg.run()
