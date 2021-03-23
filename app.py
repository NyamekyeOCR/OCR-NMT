import streamlit as st
from streamlit import caching
import easyocr
import PIL
import torch
import numpy as np
from data import *
from inputs import image_input
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

st.title("Neural machine translation on scene and written text.")
st.sidebar.title('Navigation')
method = st.sidebar.radio('Go To ->', options=['Image', 'Camera'])
st.sidebar.header('Select Image')

#style_model_name = st.sidebar.selectbox("Choose the style model: ", style_models_name)
#style_model_path = style_models_dict[style_model_name]

#model = get_model_from_path(style_model_path)

if method == 'Image':
    image_input()
