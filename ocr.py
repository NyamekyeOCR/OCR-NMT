import easyocr
import PIL
from PIL import ImageDraw
import torch


def detection(image, lang='en'):
	reader = easyocr.Reader([lang])
	bounds = reader.detect(image)
	return bounds
	

def recognition(image, lang='en', detail=True):
	
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



		
