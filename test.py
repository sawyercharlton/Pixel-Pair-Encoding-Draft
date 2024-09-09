from pixel_bpe import BasicTokenizer
from PIL import Image
from skimage.io import imread
from numpy import asarray
import torch
import torchvision
import torchvision.datasets.mnist as mnist
import os
from pixel_bpe import BasicTokenizer
from PIL import Image
from numpy import asarray
import glob

torch.set_printoptions(profile="full")  # display the full tensor list

single_image = Image.open('./data/MNIST/raw/test/1.jpg')
single_img_array = asarray(single_image)
single_img_array_flat = single_img_array.flatten()
print("pixels: ", single_img_array_flat.tolist())
print("len(img_array_flat): ", len(single_img_array_flat))

tokenizer = BasicTokenizer()
tokenizer.load('./models/basic.model')  # load trained model

output_encoder = tokenizer.encode(single_img_array_flat)
output_array = asarray(output_encoder)
print("output_array: ", output_array)
print("len(output_encoder): ", len(output_encoder), "\n", type(output_encoder))


# Valid the algorithm using decoder
output_decoder = tokenizer.decode(tokenizer.encode(output_encoder))
print("output_decoder: ", output_decoder)
# print("len(output_decoder): ", len(output_decoder))
print("pixels == output_decoder ? ", single_img_array_flat.tolist() == output_decoder)
