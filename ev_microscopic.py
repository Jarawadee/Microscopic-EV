import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
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

