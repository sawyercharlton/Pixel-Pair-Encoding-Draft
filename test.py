from image_minbpe import BasicTokenizer
from PIL import Image
from skimage.io import imread
from numpy import asarray
import torch
import torchvision
import torchvision.datasets.mnist as mnist
import os
from image_minbpe import BasicTokenizer
from PIL import Image
from numpy import asarray
import glob

torch.set_printoptions(profile="full")  # display the full tensor list

# image_tensor = train_set[0][0].view(-1)
# img_np = image_tensor.numpy()

in_dir = 'data/MNIST/raw/train'  # change this according to your dataset directory
img_list = []
img_list.extend(glob.glob(os.path.join(in_dir, "*")))
print(len(img_list))

tokenizer = BasicTokenizer()

# for in_img in img_list:
#     image = Image.open(in_img)
#     img_array = asarray(image)
#     img_array_flat = img_array.flatten()
#     # print("pixels: ", img_array_flat.tolist())
#     # print("len(img_array_flat): ", len(img_array_flat))
#     tokenizer.train(img_array_flat, 256 + 50, 2, resume=True,
#                     verbose=False)  # 256 are the byte tokens, then do ? merges

# image_1 = Image.open('./data/MNIST/raw/train/1.jpg')
# img_array_1 = asarray(image_1)
# img_array_flat_1 = img_array_1.flatten()
# print("pixels: ", img_array_flat_1.tolist())
# print("len(img_array_flat): ", len(img_array_flat_1))
#
# image_2 = Image.open('./data/MNIST/raw/train/2.jpg')
# img_array_2 = asarray(image_2)
# img_array_flat_2 = img_array_2.flatten()
# print("pixels: ", img_array_flat_2.tolist())
# print("len(img_array_flat): ", len(img_array_flat_2))

# tokenizer.save("toy")  # writes two files: toy.model (for loading) and toy.vocab (for viewing)

tokenizer.load('toy.model')  # load trained model


image_2 = Image.open('./data/MNIST/raw/test/2.jpg')
img_array_2 = asarray(image_2)
img_array_flat_2 = img_array_2.flatten()
print("pixels: ", img_array_flat_2.tolist())
print("len(img_array_flat): ", len(img_array_flat_2))

output_encoder = tokenizer.encode(img_array_flat_2)
output_array = asarray(output_encoder)
print("output_array: ", output_array)
print("len(output_encoder): ", len(output_encoder))

# valid the algorithm using decoder
# output_decoder = tokenizer.decode(tokenizer.encode(text))
# print("output_decoder: ", output_decoder)
# print("len(output_decoder): ", len(output_decoder))
# print("pixels == output_decoder ? ", img_array_flat.tolist() == output_decoder)
