import os

en_fr_ = ["Helsinki-NLP/opus-mt-en-fr"]

fr_en_ = ["Helsinki-NLP/opus-mt-fr-en"]

input_langs = {'English': 'en'}

output_langs = {'French': 'fr'}

images_path = 'images'

images_name = [x[:-4] for x in os.listdir(images_path)]

images_file = [x for x in os.listdir(images_path)]

images_dict = {name: os.path.join(images_path, filee) for name, filee in zip(images_name, images_file)}
