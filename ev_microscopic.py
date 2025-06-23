import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

#--------------------------------------------------------------------------------------------------------------
# ==== ส่วนระบบ Login ====
# ตั้ง username/password ที่อนุญาต
USERNAME = "admin"
PASSWORD = "1234"

# ตรวจสอบว่าผู้ใช้ล็อกอินแล้วหรือยัง
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ถ้ายังไม่ล็อกอิน ให้แสดงฟอร์มล็อกอิน
if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# ==== ส่วนหน้าโหวต ====
@st.dialog("Cast your vote")
def vote(item):
    st.write(f"Why is {item} your favorite?")
    reason = st.text_input("Because...")
    if st.button("Submit"):
        st.session_state.vote = {"item": item, "reason": reason}
        st.rerun()

st.title("🎉 Vote for your favorite")

if "vote" not in st.session_state:
    if st.button("A"):
        vote("A")
    if st.button("B"):
        vote("B")
else:
    st.success(f"You voted for {st.session_state.vote['item']} because {st.session_state.vote['reason']}")

#--------------------------------------------------------------------------------------------------------------
model_path = 'ev_cnn_mobile.keras'
model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

def boxlocation(img_c, box_size):
    a = b = c = d = 0
    for i in range(img_c.shape[0]):
        for j in range(img_c.shape[1]):
            if a == 0 and img_c[i, j] > 0:
                a = i
            if a != 0 and img_c[i, j] > 0:
                b = i   # != แปลว่า ไม่เท่ากับ

    for j in range(img_c.shape[1]):
        for i in range(img_c.shape[0]):
            if c == 0 and img_c[i, j] > 0:
                c = j
            if c != 0 and img_c[i, j] > 0:
                d = j


    locat = [a - box_size, b + box_size, c - box_size, d + box_size]
    return locat


def drawbox(img, label, a, b, c, d, box_size):
    image = cv2.rectangle(img, (c, a), (d, b), (255, 0, 0), 3)
    image = cv2.putText(image, label, (c + box_size, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 3)
    return image

def objectdet(img):

    box_size_y = 500
    box_size_x = 500
    step_size = 50

    img_output = np.array(img)
    img_cont = np.zeros((img_output.shape[0], img_output.shape[1]))
    result = 0
    #result = None
    #result_class = -1

    for i in range(0, img_output.shape[0] - box_size_y, step_size):
        for j in range(0, img_output.shape[1] - box_size_x, step_size):
            img_patch = img_output[i:i + box_size_y, j:j + box_size_x]
            brightness = np.mean(cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY))
            if brightness < 50:
                continue
            img_patch = cv2.resize(img_patch, (128, 128), interpolation=cv2.INTER_AREA)
            img_patch = [img_patch]
            img_patch = np.array(img_patch)

            y_outp = model.predict(img_patch, verbose=0)

            predicted_class = np.argmax(y_outp) #argmax return ค่า index ใน list ที่ให้ค่าสูงสุด
            confidence = y_outp[0][predicted_class]

           # if confidence > 0.60:
               # result = confidence
                #result_class = predicted_class
                #img_cont[i + box_size_y // 2, j + box_size_x // 2] = confidence * 255

            if result < y_outp[0][1] and y_outp[0][1] > 0.90:
                result = y_outp[0][1]
                img_cont[i+(box_size_y//2), j+(box_size_y//2)] = y_outp[0][1]*255

    boxlocat = []

    #if result is not None:
    if result != 0:
        label = "Ev egg:"+format(result, f".{2}f")
        boxlocat = boxlocation(img_cont, box_size_x // 2)
        img_output = drawbox(img, label, boxlocat[0], boxlocat[1], boxlocat[2], boxlocat[3], box_size_x // 2)

    return img_output

#--------------------------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "tif"])
if uploaded_file is not None:
    try:
        image = np.array(Image.open(uploaded_file))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        st.image(image, caption="Uploaded Image")

        output_img = objectdet(image)
        st.image(output_img, caption="Processed Image")

    except Exception as e:
        st.error(f"Error loading image: {e}")

