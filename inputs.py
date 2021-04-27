import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit import caching
import cv2
import imutils
from data import *
from ocr import detection, recognition, draw_boxes
from nmt import en_fr, fr_en
from transformers import (
		AutoTokenizer,
		AutoModelForSeq2SeqLM,
		LogitsProcessorList,
		MinLengthLogitsProcessor,
		HammingDiversityLogitsProcessor,
		BeamSearchScorer,
)

@st.cache(suppress_st_warning=True)
def image_input(recognize=True, lang='en_fr'):
    
    if st.sidebar.checkbox('Upload', value=True):
        content_file = st.sidebar.file_uploader("Choose a Content Image")
    else:
        content_name = st.sidebar.selectbox("Choose the content images:", images_name)
        content_file = images_dict[content_name]

    if content_file is not None:
        content = Image.open(content_file)
        content = np.array(content) #pil to cv
        content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR) 
    
    else:
        st.warning("Upload an Image OR Untick the Upload Button")
        
    
    # WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=200) 
    # content = imutils.resize(content, width=WIDTH)
    # generated = style_transfer(content, model)
    
    st.image(content)
    if recognize:
	    text = recognition(content)
    else:
        bounds = detection(content_file)
        content = draw_boxes(content, bounds)
    
    if lang == 'en_fr':
        translated = en_fr("Helsinki-NLP/opus-mt-en-fr", text)
    
    if lang == 'fr_en':
        translated = fr_en(fr_en_['tokenizer'], fr_en_['model'], text)
    
    trans = dict(zip([x[1] for x in text], translated))
    #st.write(translated)
    #st.write([x[1] for x in text])
    st.success(' '.join([i[-2] for i in text]))
    st.success(' '.join(translated))
   

"""
def webcam_input(model):
    st.header("Webcam Live Feed")
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([], channels='BGR')
    SIDE_WINDOW = st.sidebar.image([], width=100, channels='BGR')
    camera = cv2.VideoCapture(0)
    WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50))) 

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # orig = frame.copy()
        orig = imutils.resize(frame, width=300)
        frame = imutils.resize(frame, width=WIDTH)
        target = style_transfer(frame, model)
        FRAME_WINDOW.image(target)
        SIDE_WINDOW.image(orig)
    else:        
        st.warning("NOTE: Streamlit currently doesn't support webcam. So to use this, clone this repo and run it on local server.")
        st.warning('Stopped')
"""
