import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- 1. Config หลักของแอป (ต้องอยู่บรรทัดแรกๆ) ---
st.set_page_config(
    page_title="Pinworm Disease Diagnosis App",
    layout="wide",
    initial_sidebar_state="expanded"
)


