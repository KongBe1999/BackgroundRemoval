import streamlit as st
from PIL import Image
from io import BytesIO
import base64
from facenet_pytorch import MTCNN
import torch
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(layout="wide", page_title="Face Detection")

st.write("## Detected faces in the uploaded image")
st.write(
    ":dog: Try uploading an image to detect the faces in image. Full quality images can be downloaded from the sidebar."
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

device = torch.device("cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def detect_face(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    # fixed = remove(image)
    boxes, probs = mtcnn.detect(image)

    #draw faces
    frame_draw = image.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255,0,0), width=6)

    col2.write("Detected Image :wrench:")
    col2.image(frame_draw)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(frame_draw), "detected_image.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        detect_face(upload=my_upload)
else:
    detect_face("./crowd1.jpg")
