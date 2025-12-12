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



st.title("🔬 Pinworm Disease Diagnosis App")

st.header("ยินดีต้อนรับ!")

st.markdown("""

แอปพลิเคชันนี้ออกแบบมาเพื่อช่วยในการวินิจฉัยและให้ความรู้เกี่ยวกับ **พยาธิเข็มหมุด (_Enterobius vermicularis_)**

โปรดเลือกเมนูทางด้านซ้ายเพื่อไปยังส่วนที่ต้องการ:

""")



# Using object notation for sidebar navigation

add_selectbox = st.sidebar.selectbox(

    "เลือกเมนูการใช้งาน:",

    ("หน้าหลัก/ความรู้เกี่ยวกับพยาธิเข็มหมุด", "🔎 AI detection", "Dataset")

)



# --- 2. Model Loading (Cached for Efficiency) ---

# NOTE: The decorator must be followed by parentheses: @st.cache_resource()

@st.cache_resource()

def load_model():

    # NOTE: In a real environment, 'ev_cnn_mobile.keras' must be present in the directory.

    # For local testing, ensure the path is correct.

    model_path = 'ev_cnn_mobile.keras'

    # Added a try-except block to gracefully handle the case where the model file is not found

    try:

        model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

        return model

    except FileNotFoundError:

        st.error(f"Error: Model file not found at path '{model_path}'. Please ensure 'ev_cnn_mobile.keras' is in the current directory.")

        return None

    except Exception as e:

        st.error(f"Error loading model: {e}")

        return None



# Load the model using the cached function

model = load_model()



class_label = ["Artifact", "Ev eggs"]



def drawbox(img, label, a, b, c, d, color):
  image = cv2.rectangle(img, (c, a), (d, b), color, 2)
  image = cv2.putText(image, label, (c, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 2)
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

def ObjectDet(filepath, threshold, nms_threshold, merge_iou_threshold):
  img = cv2.imread(filepath)
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


# --- 4. Streamlit UI Flow (Section Logic) ---



if add_selectbox == "หน้าหลัก/ความรู้เกี่ยวกับพยาธิเข็มหมุด":

    st.markdown("## 📚 ความรู้เกี่ยวกับพยาธิเข็มหมุด")

    st.markdown("""

    **พยาธิเข็มหมุด (_Enterobius vermicularis_)** เป็นพยาธิที่พบบ่อยในเด็กทั่วโลก 

    

    ### ข้อมูลทั่วไป

    พยาธิตัวเมียจะวางไข่รอบๆ ทวารหนักในเวลากลางคืน ทำให้เกิดอาการคัน ไข่พยาธิมีลักษณะเฉพาะคือรูปไข่ที่ด้านหนึ่งแบน

    

    ### อาการ

    * อาการคันบริเวณทวารหนัก (โดยเฉพาะตอนกลางคืน)

    * นอนหลับไม่สนิท หงุดหงิด

    * ปวดท้องเป็นครั้งคราว หรือคลื่นไส้

    

    ### การป้องกัน

    1.  ล้างมือให้สะอาดก่อนรับประทานอาหารและหลังเข้าห้องน้ำ

    2.  ตัดเล็บให้สั้นเพื่อป้องกันการสะสมของไข่พยาธิ

    3.  ซักเครื่องนอนและเสื้อผ้าด้วยน้ำร้อนเป็นประจำ

    

    **คำเตือน:** ข้อมูลนี้ใช้เพื่อการศึกษาเท่านั้น โปรดปรึกษาแพทย์หรือผู้เชี่ยวชาญทางการแพทย์สำหรับการวินิจฉัยและการรักษาที่ถูกต้อง

    """)



elif add_selectbox == "🔎 AI detection":

    st.markdown("## 🔎 AI Detection (การตรวจหาพยาธิเข็มหมุด)")

    st.markdown("โปรดอัปโหลดภาพจากกล้องจุลทรรศน์ของการตรวจหาไข่พยาธิ (Tape Test/Swab Test) เพื่อให้ AI ทำการวิเคราะห์")

    

    # 📌 Define default parameter values for the ObjectDet function

    DEFAULT_THRESHOLD = 0.5

    DEFAULT_NMS_THRESHOLD = 0.3

    DEFAULT_MERGE_IOU_THRESHOLD = 0.2

    

    # Optional: Allow user to adjust parameters in the sidebar

    with st.sidebar.expander("⚙️ ปรับค่าพารามิเตอร์ AI (ขั้นสูง)"):

        detection_threshold = st.slider("Detection Threshold (ความเชื่อมั่นขั้นต่ำ)", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05)

        nms_threshold = st.slider("NMS IOU Threshold (การตัดกล่องที่ซ้ำซ้อน)", 0.0, 1.0, DEFAULT_NMS_THRESHOLD, 0.05)

        merge_iou_threshold = st.slider("Merge IOU Threshold (การรวมกล่องใกล้เคียง)", 0.0, 1.0, DEFAULT_MERGE_IOU_THRESHOLD, 0.05)

    

    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (PNG, JPG, JPEG, TIF)", type=["png", "jpg", "jpeg", "tif"])

    

    if uploaded_file is not None:

        try:

            # Read the file from the uploader

            image = Image.open(uploaded_file)

            image_np = np.array(image.convert("RGB")) # Ensure it's 3-channel (RGB)

            

            # Convert RGB to BGR for OpenCV processing (mandatory for cv2 functions)

            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)



            col1, col2 = st.columns(2)

            

            with col1:

                st.subheader("ภาพต้นฉบับ")

                st.image(image_np, caption=uploaded_file.name, use_column_width=True)



            if model is not None: # Check if model loaded successfully

                # Perform detection - ❗ IMPORTANT: Pass the required arguments

                with st.spinner('กำลังวิเคราะห์ภาพด้วย AI...'):

                    output_img_bgr = ObjectDet(image_bgr, detection_threshold, nms_threshold, merge_iou_threshold)

                

                # Convert the result back to RGB for Streamlit display

                output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)

                

                with col2:

                    st.subheader("ผลการวิเคราะห์")

                    st.image(output_img_rgb, caption="ภาพพร้อมกล่องระบุไข่พยาธิ (ถ้าพบ)", use_column_width=True)

            else:

                with col2:

                    st.subheader("ผลการวิเคราะห์")

                    # Warning handled by load_model function, but this acts as a fallback

                    st.warning("ไม่สามารถทำการวิเคราะห์ได้ เนื่องจากโมเดล AI โหลดไม่สำเร็จ กรุณาตรวจสอบไฟล์โมเดล ('ev_cnn_mobile.keras').")



        except Exception as e:

            st.error(f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {e}")
