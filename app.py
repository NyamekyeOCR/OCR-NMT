import streamlit as st
from streamlit import caching
import easyocr
from PIL import (
    Image,
    ImageDraw
)
import torch
import numpy as np
from data import *
import cv2
import imutils
from transformers import (
	AutoTokenizer,
	AutoModelForSeq2SeqLM,
	LogitsProcessorList,
	MinLengthLogitsProcessor,
	HammingDiversityLogitsProcessor,
	BeamSearchScorer,
)

def main():
    st.title("NMT on scene and written text.")
    st.sidebar.title('Features')
    st.sidebar.header('Select Image Source')
    method = st.sidebar.radio('Go To ->', options=['Image', 'Camera'])
    st.sidebar.header('Select Image')
    submit = st.sidebar.button('Submit')
    

    if method == 'Image':
        if st.sidebar.checkbox('Upload', value=True):
            content_file = st.sidebar.file_uploader("Choose a Content Image")
        else:
            content_name = st.sidebar.selectbox("Choose the content images:", images_name)
            content_file = images_dict[content_name]

        if content_file is not None:
            content = Image.open(content_file)
            content = np.array(content) #pil to cv
            content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR) 

            st.image(content)
        
        else:
            st.warning("Upload an Image OR Untick the Upload Button")

    else:
        pass 


    if st.sidebar.checkbox('Show advanced options'):
        detect_rec = st.sidebar.radio('Go To ->', options=['Recognition', 'Detection'])
        ip_lang = st.sidebar.selectbox("Choose the input language:", [keys for keys in input_langs.keys()])
        op_lang = st.sidebar.selectbox("Choose the output images:", [keys for keys in output_langs.keys()])   
        if submit:
            if detect_rec == 'Recognition':
                text = recognition(content, input_langs[ip_lang])
            else:
                bounds = detection(content_file, input_langs[ip_lang])
                content = draw_boxes(content, bounds)

            if ip_lang:
                translated = en_fr("Helsinki-NLP/opus-mt-en-fr", text)
        
            
            trans = dict(zip([x[1] for x in text], translated))
            #st.write(translated)
            #st.write([x[1] for x in text])
            st.success(' '.join([i[-2] for i in text]))
            st.success(' '.join(translated))
    
   

    

def detection(image, lang):
	reader = easyocr.Reader([lang])
	bounds = reader.detect(image)
	return bounds
	

def recognition(image, lang, detail=True):
	
	reader = easyocr.Reader([lang])
	if detail:
		bounds = reader.readtext(image)
	else:
		bounds = reader.readtext(image, detail=0)
	return bounds
	

def draw_boxes(image, bounds, color='red', width=2):
	draw = PIL.ImageDraw.Draw(image)
	for bound in bounds:
		x0, x1, x2, x3 = bound[0]
		draw.line([*x0, *x1, *x2, *x3, *x0], fill=color, width=width)
	return image

def en_fr(model_name, src_text, num_beams=6):

	src_text = [x[1] for x in src_text]

	model_name = model_name

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

	translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))

	tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

	return tgt_text



def fr_en(model_name, text, num_beams=6):

	src_text = [x[1] for x in src_text]

	model_name = model_name

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

	translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))

	tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

	return tgt_text


if __name__ == '__main__':
    main()





#style_model_name = st.sidebar.selectbox("Choose the style model: ", style_models_name)
#style_model_path = style_models_dict[style_model_name]

#model = get_model_from_path(style_model_path)
