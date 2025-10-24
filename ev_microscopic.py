import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
'''
st.set_page_config(
    page_title="Pinworm Disease Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🔬 Pinworm Disease Diagnosis App")
st.header("ยินดีต้อนรับ!")
st.markdown("""
แอปพลิเคชันนี้ออกแบบมาเพื่อช่วยในการวินิจฉัยและให้ความรู้เกี่ยวกับ **พยาธิเข็มหมุด (_Enterobius vermicularis_)**
โปรดเลือกเมนูทางด้านซ้ายเพื่อไปยังส่วนที่ต้องการ:
* **📚 ความรู้เกี่ยวกับพยาธิเข็มหมุด:** ข้อมูลทั่วไป, อาการ, และการป้องกัน
* **🔎 AI detection:** อัปโหลดภาพจากกล้องจุลทรรศน์เพื่อตรวจหาไข่พยาธิ
""")
# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("ความรู้เกี่ยวกับพยาธิเข็มหมุด", "AI detection")
)

#--------------------------------------------------------------------------------------------------------------
model_path = 'ev_cnn_mobile.keras'
model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    
class_label = ["Artifact", "Ev eggs"]

def drawbox(img, label, a, b, c, d, color):
  image = cv2.rectangle(img, (c, a), (d, b), (255, 0, 0), 3)
  image = cv2.putText(image, label, (c, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 3)
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
      if brightness < 50:
        continue
      img_patch = cv2.resize(img_patch, (resize_input_y, resize_input_x), interpolation=cv2.INTER_AREA)
      patches.append(img_patch)
      coords.append((i, j))

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
  colors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0)]
  for det in merged_detections:
    a, b, c, d = det['bbox']
    class_idx = det['class_idx']
    label = f"{class_label[class_idx]}: {det['score']:.2f}"
    color = colors[class_idx % len(colors)]
    img_output = drawbox(img_output, label, a, b, c, d, color)
  return img_output
#--------------------------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "tif"])
if uploaded_file is not None:
    try:
        image = np.array(Image.open(uploaded_file))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        st.image(image, caption="Uploaded Image")

        output_img = ObjectDet(image, 0.99, 0.2, 0.3)
        st.image(output_img, caption="Processed Image")

    except Exception as e:
        st.error(f"Error loading image: {e}")
'''

# -------------------------------------------------------------
# 🧩 CONFIG PAGE
# -------------------------------------------------------------
st.set_page_config(
    page_title="Pinworm Disease Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Pinworm Disease Diagnosis App")
st.header("ยินดีต้อนรับ!")

st.markdown("""
แอปพลิเคชันนี้ออกแบบมาเพื่อช่วยในการวินิจฉัยและให้ความรู้เกี่ยวกับ  
**พยาธิเข็มหมุด (_Enterobius vermicularis_)** 🪱  
โดยใช้เทคนิคปัญญาประดิษฐ์ (AI) เพื่อช่วยตรวจหาภาพไข่พยาธิจากกล้องจุลทรรศน์
""")

# -------------------------------------------------------------
# 🧭 SIDEBAR MENU
# -------------------------------------------------------------
add_selectbox = st.sidebar.radio(
    "เลือกเมนูที่ต้องการ:",
    ("📚 ความรู้เกี่ยวกับพยาธิเข็มหมุด", "🤖 AI detection")
)

# -------------------------------------------------------------
# 📘 KNOWLEDGE SECTION
# -------------------------------------------------------------
if add_selectbox == "📚 ความรู้เกี่ยวกับพยาธิเข็มหมุด":
    st.subheader("🦠 ข้อมูลทั่วไปเกี่ยวกับพยาธิเข็มหมุด (_Enterobius vermicularis_)")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/2d/Enterobius_vermicularis_life_cycle.png",
        caption="วงจรชีวิตของพยาธิเข็มหมุด (CDC, 2023)",
        use_container_width=True
    )

    st.markdown("""
### 🔍 ลักษณะทั่วไป  
- พยาธิเข็มหมุดเป็นพยาธิตัวกลมขนาดเล็ก สีขาว ยาวประมาณ 0.5 – 1 เซนติเมตร  
- พบได้บ่อยในเด็ก โดยเฉพาะในพื้นที่ที่สุขาภิบาลไม่ดี  
- ไข่ของพยาธิมีลักษณะใส รูปรี และสามารถติดตามสิ่งของหรือเล็บมือได้ง่าย  

### ⚠️ อาการที่พบบ่อย  
- คันบริเวณทวารหนัก โดยเฉพาะตอนกลางคืน  
- นอนไม่หลับ หงุดหงิด  
- อาจพบไข่พยาธิในอุจจาระหรือบริเวณก้น  

### 🧫 การวินิจฉัย  
- วิธีมาตรฐานคือการตรวจด้วย **cellophane tape test** (การใช้เทปใสแตะบริเวณรอบทวารหนักตอนเช้า)  
- การตรวจด้วยกล้องจุลทรรศน์สามารถเห็นไข่พยาธิรูปรีมีฝาแบนด้านหนึ่ง  

### 🧍‍♀️ การป้องกัน  
- ล้างมือให้สะอาดก่อนรับประทานอาหารและหลังเข้าห้องน้ำ  
- ตัดเล็บให้สั้น ไม่แคะก้น  
- ซักทำความสะอาดผ้าปูที่นอนและเสื้อผ้าบ่อย ๆ  
- รักษาทุกคนในครอบครัวพร้อมกันเมื่อพบการติดเชื้อ  

### 💊 การรักษา  
- ใช้ยาถ่ายพยาธิ เช่น **Mebendazole** หรือ **Albendazole** ตามคำแนะนำของแพทย์  
- ควรกินซ้ำอีกครั้งหลังจาก 2 สัปดาห์เพื่อฆ่าพยาธิที่ฟักใหม่  
    """)

    st.info("💡 เคล็ดลับ: การป้องกันดีกว่าการรักษา — รักษาความสะอาดและล้างมือเสมอ!")

# -------------------------------------------------------------
# 🤖 AI DETECTION SECTION
# -------------------------------------------------------------
elif add_selectbox == "🤖 AI detection":
    st.subheader("🧠 ระบบ AI สำหรับการตรวจจับไข่พยาธิเข็มหมุด")

    st.markdown("""
    อัปโหลดภาพจากกล้องจุลทรรศน์เพื่อให้ระบบ AI วิเคราะห์  
    ระบบนี้ถูกพัฒนาโดยใช้ **Convolutional Neural Network (CNN)**  
    เพื่อจำแนกภาพว่าเป็น  
    - **Ev eggs (ไข่พยาธิเข็มหมุด)** หรือ  
    - **Artifact (สิ่งแปลกปลอม)**

    **ขั้นตอนการใช้งาน:**
    1. อัปโหลดไฟล์ภาพ (.jpg, .png, .tif)
    2. รอระบบประมวลผล
    3. ภาพผลลัพธ์จะแสดงตำแหน่งไข่พยาธิที่ตรวจพบ  
    """)

    # โหลดโมเดล
    model_path = 'ev_cnn_mobile.keras'
    model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    class_label = ["Artifact", "Ev eggs"]

    # ---------------------- ฟังก์ชัน ----------------------
    def drawbox(img, label, a, b, c, d, color):
        image = cv2.rectangle(img, (c, a), (d, b), color, 3)
        image = cv2.putText(image, label, (c, a - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
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
        return inter_area / union_area if union_area else 0

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
                if brightness < 50:
                    continue
                img_patch = cv2.resize(img_patch, (resize_input_y, resize_input_x))
                patches.append(img_patch)
                coords.append((i, j))
        if not patches:
            return img
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
        merged_detections = merge_connected_boxes_by_class(nms_detections, merge_iou_threshold)
        img_output = img.copy()
        colors = [(0,255,0), (255,0,0), (0,0,255)]
        for det in merged_detections:
            a, b, c, d = det['bbox']
            label = f"{class_label[det['class_idx']]}: {det['score']:.2f}"
            img_output = drawbox(img_output, label, a, b, c, d, colors[det['class_idx'] % len(colors)])
        return img_output

    # ---------------------- Upload Section ----------------------
    uploaded_file = st.file_uploader("📤 อัปโหลดภาพจากกล้องจุลทรรศน์", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        try:
            image = np.array(Image.open(uploaded_file))
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            st.image(image, caption="📸 ภาพที่อัปโหลด", use_container_width=True)
            st.info("🔎 กำลังประมวลผลด้วยโมเดล AI ...")
            output_img = ObjectDet(image, 0.99, 0.2, 0.3)
            st.image(output_img, caption="✅ ผลการตรวจจับ", use_container_width=True)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างโหลดภาพ: {e}")


