import streamlit as st

# ตั้งค่า Config หลักของแอปที่นี่ครั้งเดียว
st.set_page_config(
    page_title="Pinworm Diagnosis App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- กำหนดหน้าย่อยๆ ---

# กลุ่มความรู้
info_1 = st.Page("ev_microscopic.py/1_general.py", title="ข้อมูลทั่วไป", icon="📄")
info_2 = st.Page("ev_microscopic.py/2_symptoms.py", title="อาการ", icon="🤒")
info_3 = st.Page("ev_microscopic.py/3_prevention.py", title="การป้องกัน", icon="🛡️")

# กลุ่มเครื่องมือ
tool_ai = st.Page("ev_microscopic.py/ai_detect.py", title="AI Detection", icon="🔎")
tool_data = st.Page("ev_microscopic.py/dataset.py", title="Dataset", icon="📊")

# --- สร้างระบบนำทาง (Navigation) ---
# จัดกลุ่มหน้าให้อยู่ภายใต้หัวข้อ
pg = st.navigation({
    "ความรู้เกี่ยวกับพยาธิเข็มหมุด": [info_1, info_2, info_3],
    "ระบบวินิจฉัยและข้อมูล": [tool_ai, tool_data]
})

# รันหน้าปัจจุบัน
pg.run()
