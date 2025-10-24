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
    ("หน้าหลัก/ความรู้เกี่ยวกับพยาธิเข็มหมุด", "🔎 AI detection")
)

# --- 2. Model Loading (Cached for Efficiency) ---
# NOTE: In a real environment, 'ev_cnn_mobile.keras' must be present in the directory.
# Since we are in an execution environment without guaranteed external files, 
# we must assume the path is correct for local testing.
@st.cache_resource
def load_detection_model():
    """Loads the TensorFlow/Keras model using Streamlit's resource cache."""
    try:
        # Assuming the model is named 'ev_cnn_mobile.keras'
        model_path = 'ev_cnn_mobile.keras' 
        # We use compile=False if the model is already saved with weights and architecture
        # and we don't need to retrain or recompile the graph.
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        # If the model file is not found, display an error and return None
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the 'ev_cnn_mobile.keras' file is in the application directory.")
        return None

model = load_detection_model()
class_label = ["Artifact", "Ev eggs"]
patch_sizes = {1: (500, 500)} # Only targeting Ev eggs with index 1
step_size = 50

# --- 3. Object Detection Function ---

def drawbox(img, label, a, b, c, d, box_size):
    """Draws a bounding box and label on the image."""
    # Ensure coordinates are within image boundaries
    a = max(0, a)
    b = min(img.shape[0], b)
    c = max(0, c)
    d = min(img.shape[1], d)

    # Use BGR colors for OpenCV
    image = cv2.rectangle(img, (c, a), (d, b), (0, 255, 0), 5) # Green box, thicker
    
    # Use a smaller font size for better visibility
    font_scale = 1.5
    image = cv2.putText(image, label, (c, a - 10), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 0, 255), 2)
    return image

def ObjectDet(img):
    """
    Performs sliding window object detection on the input image array.
    
    Args:
        img (np.array): The input image as a NumPy array (H, W, C).
    
    Returns:
        np.array: The image with detected boxes drawn on it.
    """
    if model is None:
        return img # Return original image if model failed to load
        
    img_output = np.array(img)
    img_height, img_width = img_output.shape[:2]
    
    # Track the best result found for each class based on the highest score
    # We only care about class_idx 1 (Ev eggs) based on patch_sizes definition
    best_results = [{'score': 0, 'loc': None, 'patch_size': None} for _ in class_label]
    
    detection_found = False

    # Since patch_sizes only contains key 1, we iterate only over that.
    # We assume class_idx 1 corresponds to "Ev eggs" based on class_label
    class_idx = 1 
    box_size_y, box_size_x = patch_sizes[class_idx]
    
    # Sliding window logic
    for i in range(0, img_height - box_size_y + 1, step_size):
        for j in range(0, img_width - box_size_x + 1, step_size):
            img_patch = img_output[i:i+box_size_y, j:j+box_size_x]
            
            # Skip if the patch is incomplete (shouldn't happen with range setup, but safe check)
            if img_patch.shape[0] != box_size_y or img_patch.shape[1] != box_size_x:
                continue

            # Pre-process the patch for the 64x64 input model
            img_patch_resized = cv2.resize(img_patch, (64, 64), interpolation=cv2.INTER_AREA)
            img_patch_resized = np.expand_dims(img_patch_resized, axis=0)
            
            # Predict
            try:
                # Normalize image patch if the model was trained on normalized data (0-1 or -1 to 1)
                # Assuming the model expects values in 0-255 based on original code style.
                y_outp = model.predict(img_patch_resized, verbose=0)
                y_pred = y_outp[0]
            except Exception as e:
                st.warning(f"Prediction error: {e}")
                continue

            # Get the score for the target class (Ev eggs)
            score = y_pred[class_idx]
            
            # Check if this patch is better than the current best result and meets the high confidence threshold
            if score > best_results[class_idx]['score'] and score > 0.95:
                best_results[class_idx] = {'score': score, 'loc': (i, j), 'patch_size': (box_size_y, box_size_x)}
                detection_found = True

    # Draw the best bounding box(es) found
    if detection_found:
        result = best_results[class_idx]
        
        label = f"{class_label[class_idx]}: {result['score']:.2f}"
        i, j = result['loc']
        box_size_y, box_size_x = result['patch_size']
        
        # Calculate bounding box coordinates
        a, b, c, d = i, i+box_size_y, j, j+box_size_x
        
        # Draw the box on the output image
        img_output = drawbox(img_output, label, a, b, c, d, box_size_x//2)
    else:
        st.info("No Pinworm eggs detected with high confidence (score > 0.95).")

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

            if model:
                # Perform detection
                with st.spinner('กำลังวิเคราะห์ภาพด้วย AI...'):
                    output_img_bgr = ObjectDet(image_bgr)
                
                # Convert the result back to RGB for Streamlit display
                output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("ผลการวิเคราะห์")
                    st.image(output_img_rgb, caption="ภาพพร้อมกล่องระบุไข่พยาธิ (ถ้าพบ)", use_column_width=True)
            else:
                 with col2:
                    st.subheader("ผลการวิเคราะห์")
                    st.warning("ไม่สามารถทำการวิเคราะห์ได้ เนื่องจากโมเดล AI โหลดไม่สำเร็จ กรุณาตรวจสอบไฟล์โมเดล")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {e}")
            st.error("โปรดตรวจสอบว่าไฟล์รูปภาพของคุณเสียหายหรือไม่")
            st.code(e, language='python') # Show error for debugging

