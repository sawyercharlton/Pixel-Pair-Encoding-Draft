from image_minbpe import BasicTokenizer
from PIL import Image
from numpy import asarray

image = Image.open('./0.jpg')
img_np = asarray(image)
img_np_flat = img_np.flatten()
print(img_np_flat)
tokenizer = BasicTokenizer()

text = img_np_flat
tokenizer.train(text, 256 + 48)  # 256 are the byte tokens, then do 50 merges
output_encoder = tokenizer.encode(text)
print(output_encoder)
output_decoder = tokenizer.decode(tokenizer.encode(text))
print(output_decoder)
tokenizer.save("toy")
# writes two files: toy.model (for loadinrg) and toy.vocab (for viewing)