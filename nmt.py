import torch
from transformers import (
	AutoTokenizer,
	AutoModelForSeq2SeqLM,
	LogitsProcessorList,
	MinLengthLogitsProcessor,
	HammingDiversityLogitsProcessor,
	BeamSearchScorer,
)



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
