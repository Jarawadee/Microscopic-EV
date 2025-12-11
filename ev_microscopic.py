import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import io

# --- 1. Streamlit Configuration & Custom Styles ---
st.set_page_config(
    page_title="Pinworm Disease Diagnosis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .stAlert { border-radius: 10px; }
    .reportview-container .main .block-container{ max-width: 1200px; }
</style>
""", unsafe_allow_html=True)

# --- 2. Model Loading (Cached) ---
@st.cache_resource()
def load_model():
    model_path = 'ev_cnn_mobile.keras'
    try:
        # Load model with custom objects if needed (e.g., losses)
        model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        return model
    except FileNotFoundError:
        st.error(f"⚠️ Error: Model file '{model_path}' not found. Please upload the model file.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# --- 3. Helper Functions (Detection Logic) ---
class_label = ["Artifact", "Ev eggs"]

def drawbox(img, label, a, b, c, d, color):
    # Draw thicker box for visibility
    image = cv2.rectangle(img, (c, a), (d, b), color, 4) 
    # Add background for text to make it readable
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    image = cv2.rectangle(image, (c, a - 30), (c + w, a), color, -1)
    image = cv2.putText(image, label, (c, a - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
                    if j not in used:
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
    if model is None: return img.copy(), 0 # Return 0 count

    box_size_y, box_size_x, step_size = 500, 500, 50
    resize_input_y, resize_input_x = 64, 64
    img_h, img_w = img.shape[:2]

    coords = []
    patches = []
    for i in range(0, img_h - box_size_y + 1, step_size):
        for j in range(0, img_w - box_size_x + 1, step_size):
            img_patch = img[i:i+box_size_y, j:j+box_size_x]
            if np.mean(cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)) < 50: continue
            img_patch = cv2.resize(img_patch, (resize_input_y, resize_input_x), interpolation=cv2.INTER_AREA)
            patches.append(img_patch)
            coords.append((i, j))
    
    if not patches:
        return img.copy(), 0

    patches = np.array(patches)
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
        merged_detections = merge_connected_boxes_by_class(nms_detections, merge_iou_threshold=merge_iou_threshold)
    else:
        merged_detections = nms_detections

    img_output = img.copy()
    # Colors: Green for Artifact, Red for Eggs (assuming index 1 is eggs)
    colors = [(0,255,0), (0,0,255)] 
    
    egg_count = 0
    for det in merged_detections:
        a, b, c, d = det['bbox']
        class_idx = det['class_idx']
        
        # Count only if it's an egg (index 1)
        if class_idx == 1:
            egg_count += 1
            
        label = f"{class_label[class_idx]} ({det['score']:.2f})"
        color = colors[class_idx % len(colors)]
        img_output = drawbox(img_output, label, a, b, c, d, color)
        
    return img_output, egg_count

# --- 4. Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063216.png", width=100)
st.sidebar.title("เมนูการใช้งาน")
add_selectbox = st.sidebar.radio(
    "",
    ("🏠 หน้าหลัก & ความรู้", "🔎 AI Diagnosis", "📊 Dataset Info")
)

st.sidebar.info(
    """
    **เกี่ยวกับแอป:**
    แอปพลิเคชันนี้ใช้ Deep Learning (CNN) 
    ในการตรวจจับไข่พยาธิเข็มหมุดจากภาพถ่ายจุลทรรศน์
    """
)

# --- 5. Main Content Flow ---

if add_selectbox == "🏠 หน้าหลัก & ความรู้":
    st.title("🔬 Pinworm Disease Diagnosis App")
    st.markdown("### ระบบช่วยวินิจฉัยและให้ความรู้โรคพยาธิเข็มหมุด")
    st.divider()

    # Use Tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🐛 รู้จักพยาธิเข็มหมุด", "🩺 การตรวจและรักษา", "🛡️ การป้องกัน"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("พยาธิเข็มหมุดคืออะไร?")
            st.markdown("""
            **พยาธิเข็มหมุด (_Enterobius vermicularis_)** เป็นพยาธิตัวกลมขนาดเล็ก สีขาว คล้ายเส้นด้าย พบบ่อยในเด็กทั่วโลก
            
            **อาการที่พบบ่อย:**
            * 😰 **คันทวารหนัก:** อาการเด่นชัดที่สุด มักเกิดในเวลากลางคืน เพราะพยาธิตัวเมียจะคลานออกมาวางไข่
            * 😴 **นอนไม่หลับ:** เด็กอาจงอแง พลิกตัวไปมา หรือนอนกัดฟัน
            * 🤢 **อาการทางเดินอาหาร:** ปวดท้อง คลื่นไส้ เบื่ออาหาร (ในรายที่มีพยาธิมาก)
            """)
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Enterobius_vermicularis_female.JPG/300px-Enterobius_vermicularis_female.JPG", 
                     caption="ลักษณะพยาธิตัวเมีย (อ้างอิง: Wikimedia)")

    with tab2:
        st.subheader("วิธีการตรวจหาไข่พยาธิ (Scotch Tape Technique)")
        st.info("💡 วิธีนี้เป็นวิธีมาตรฐานและแม่นยำที่สุด (และเป็นภาพที่ AI นี้ใช้ในการวิเคราะห์)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. ช่วงเวลา")
            st.write("ควรทำใน **ตอนเช้าหลังตื่นนอนทันที** ก่อนเข้าห้องน้ำหรืออาบน้ำ")
        with c2:
            st.markdown("#### 2. วิธีการแปะ")
            st.write("ใช้เทปใสแปะที่บริเวณรูทวารหนักหลายๆ ครั้ง เพื่อให้ไข่พยาธิติดมากับเทป")
        with c3:
            st.markdown("#### 3. การส่งตรวจ")
            st.write("นำเทปใสแปะลงบนสไลด์แก้ว แล้วนำไปส่องกล้องจุลทรรศน์ (หรือถ่ายรูปมาให้ AI วิเคราะห์)")

    with tab3:
        st.subheader("การรักษาและการป้องกัน")
        st.success("""
        **การรักษา:**
        * ทานยาถ่ายพยาธิตามแพทย์สั่ง (เช่น Mebendazole หรือ Albendazole)
        * **สำคัญ:** ควรทานยาทั้งครอบครัวพร้อมกัน เพราะโรคนี้ติดต่อกันง่ายมาก
        
        **การป้องกัน:**
        1.  🛁 ตัดเล็บให้สั้นและล้างมือบ่อยๆ
        2.  ☀️ นำที่นอนหมอนมุ้งไปตากแดดจัด
        3.  👖 ซักกางเกงในและชุดนอนด้วยน้ำร้อน
        """)

elif add_selectbox == "🔎 AI Diagnosis":
    st.title("🔎 AI Detection (วิเคราะห์ภาพถ่าย)")
    
    col_upload, col_settings = st.columns([3, 1])
    
    with col_settings:
        st.subheader("⚙️ ตั้งค่า AI")
        with st.expander("ปรับค่าพารามิเตอร์ (Advanced)", expanded=False):
            detection_threshold = st.slider("Conf. Threshold", 0.0, 1.0, 0.5, 0.05, help="ความมั่นใจขั้นต่ำของ AI")
            nms_threshold = st.slider("NMS Threshold", 0.0, 1.0, 0.3, 0.05, help="ค่าการซ้อนทับสูงสุดที่ยอมรับได้")
            merge_iou_threshold = st.slider("Merge Threshold", 0.0, 1.0, 0.2, 0.05, help="รวมกล่องที่อยู่ใกล้กัน")

    with col_upload:
        st.info("📸 **คำแนะนำ:** อัปโหลดภาพจากกล้องจุลทรรศน์ที่ได้จากการทำ Scotch Tape Technique")
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg", "tif"])

    if uploaded_file is not None:
        st.divider()
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🖼️ ภาพต้นฉบับ")
                st.image(image_np, use_column_width=True, caption=f"File: {uploaded_file.name}")

            if model is not None:
                with st.spinner('🤖 AI กำลังสแกนหาไข่พยาธิ...'):
                    # Call ObjectDet and get both image and count
                    output_img_bgr, egg_count = ObjectDet(image_bgr, detection_threshold, nms_threshold, merge_iou_threshold)
                
                output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
                
                with c2:
                    st.markdown("##### ✅ ผลการวิเคราะห์")
                    st.image(output_img_rgb, use_column_width=True, caption="ผลลัพธ์การตรวจจับ")
                
                # Show Metrics Below
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("สถานะโมเดล", "Ready", delta_color="normal")
                m2.metric("จำนวนไข่พยาธิที่พบ (Ev eggs)", f"{egg_count} ฟอง", delta=egg_count, delta_color="inverse")
                
                if egg_count > 0:
                    st.warning(f"⚠️ **ตรวจพบไข่พยาธิ {egg_count} จุด** โปรดปรึกษาแพทย์เพื่อรับการรักษา")
                else:
                    st.success("✅ **ไม่พบไข่พยาธิในภาพนี้** (หรือความชัดเจนไม่เพียงพอ)")
            
            else:
                st.error("Model Error: ไม่สามารถโหลดโมเดลได้")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

elif add_selectbox == "📊 Dataset Info":
    st.title("📊 เกี่ยวกับชุดข้อมูล (Dataset)")
    st.markdown("""
    โมเดล AI นี้ถูกเทรนด้วยชุดข้อมูลภาพถ่ายจากกล้องจุลทรรศน์ โดยแบ่งคลาสออกเป็น:
    
    1.  **Ev eggs (Enterobius vermicularis eggs):** ไข่ของพยาธิเข็มหมุด มีลักษณะรี ด้านหนึ่งนูน อีกด้านหนึ่งแบน (D-shaped)
    2.  **Artifacts:** สิ่งรบกวนในภาพ เช่น ฟองอากาศ, เส้นใยผ้า, หรือฝุ่นละออง ซึ่ง AI ถูกสอนให้แยกแยะออกจากไข่พยาธิ
    """)
    
    st.image("https://www.cdc.gov/dpdx/enterobiasis/modules/Enterobius_eggs_montage.jpg", caption="ตัวอย่างลักษณะไข่พยาธิ (Credit: CDC)")
    
    st.info("""
    **หมายเหตุ:** ความแม่นยำของโมเดลขึ้นอยู่กับคุณภาพของรูปภาพ แสงสว่าง และกำลังขยายของกล้องจุลทรรศน์
    """)

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Developed for Medical Diagnosis Support | Powered by TensorFlow & Streamlit</div>", unsafe_allow_html=True)
            
