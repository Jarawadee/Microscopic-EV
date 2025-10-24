import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

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

patch_sizes = {1: (500, 500)}
step_size = 50

def boxlocation(img_c, box_size):
  a = b = c = d = 0
  for i in range(img_c.shape[0]):
    for j in range(img_c.shape[1]):
      if a==0 and img_c[i,j]>0:
        a = i
      if a!=0 and img_c[i,j]>0:
        b = i
  for j in range(img_c.shape[1]):
    for i in range(img_c.shape[0]):
      if c==0 and img_c[i,j]>0:
        c = j
      if c!=0 and img_c[i,j]>0:
        d = j
  locat = [a-box_size, b+box_size, c-box_size, d+box_size]
  return locat

def drawbox(img, label, a, b, c, d, box_size):
  image = cv2.rectangle(img, (c, a), (d, b), (0, 255, 0), 2)
  image = cv2.putText(image, label, (c + box_size, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 1)
  return image

def ObjectDet(filepath):
  img = cv2.imread(filepath)
  img_output = np.array(img)
  img_height, img_width = img_output.shape[:2]
  img_cont = np.zeros((img_height, img_width, len(class_label)))
  best_results = [{'score': 0, 'loc': None, 'patch_size': None} for _ in class_label]

  for class_idx in patch_sizes.keys():
    box_size_y, box_size_x = patch_sizes[class_idx]
    for i in range(0, img_height - box_size_y + 1, step_size):
      for j in range(0, img_width - box_size_x + 1, step_size):
        img_patch = img_output[i:i+box_size_y, j:j+box_size_x]
        if img_patch.shape[0] != box_size_y or img_patch.shape[1] != box_size_x:
          continue

        img_patch_resized = cv2.resize(img_patch, (64, 64), interpolation=cv2.INTER_AREA)
        img_patch_resized = np.expand_dims(img_patch_resized, axis=0)
        y_outp = model.predict(img_patch_resized, verbose=0)
        y_pred = y_outp[0]

        score = y_pred[class_idx]
        if score > best_results[class_idx]['score'] and score > 0.95:
          best_results[class_idx] = {'score': score, 'loc': (i, j), 'patch_size': (box_size_y, box_size_x)}
          img_cont[i + box_size_y // 2, j + box_size_x // 2, class_idx] = score * 255

  for class_idx, result in enumerate(best_results):
    if result['score'] > 0:
      label = f"{class_label[class_idx]}:{result['score']:.2f}"
      i, j = result['loc']
      box_size_y, box_size_x = result['patch_size']
      a, b, c, d = i, i+box_size_y, j, j+box_size_x
      img_output = drawbox(img_output, label, a, b, c, d, box_size_x//2)

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

