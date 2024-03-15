from image_minbpe import BasicTokenizer
from PIL import Image
from numpy import asarray

image = Image.open('./0.jpg')
img_np = asarray(image)
img_np_flat = img_np.flatten()
tokenizer = BasicTokenizer()

text = img_np_flat
tokenizer.train(text, 256 + 50) # 256 are the byte tokens, then do 50 merges
print(tokenizer.encode(text))
# print(list(tokenizer.decode(tokenizer.encode(text))))
tokenizer.save("toy")
# writes two files: toy.model (for loadinrg) and toy.vocab (for viewing)