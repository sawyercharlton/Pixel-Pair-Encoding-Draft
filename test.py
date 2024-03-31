from image_minbpe import BasicTokenizer
from PIL import Image
from numpy import asarray
import torch
import torchvision
import torchvision.datasets.mnist as mnist
import os
from image_minbpe import BasicTokenizer
from PIL import Image
from numpy import asarray

torch.set_printoptions(profile="full")  # display the full tensor list

# image_tensor = train_set[0][0].view(-1)
# img_np = image_tensor.numpy()
image = Image.open('./data/MNIST/raw/train/2.jpg')
img_array = asarray(image)
img_array_flat = img_array.flatten()
print("pixels: ", img_array_flat.tolist())
print("len(img_array_flat): ", len(img_array_flat))

tokenizer = BasicTokenizer()

text = img_array_flat
# tokenizer.train(text, 256 + 50, 3, verbose=True)  # 256 are the byte tokens, then do ? merges
# tokenizer.save("toy")  # writes two files: toy.model (for loading) and toy.vocab (for viewing)

tokenizer.load('toy.model')  # load trained model

output_encoder = tokenizer.encode(text)
output_array = asarray(output_encoder)
print("output_array: ", output_array)
print("len(output_encoder): ", len(output_encoder))

# valid the algorithm using decoder
# output_decoder = tokenizer.decode(tokenizer.encode(text))
# print("output_decoder: ", output_decoder)
# print("len(output_decoder): ", len(output_decoder))
# print("pixels == output_decoder ? ", img_array_flat.tolist() == output_decoder)
