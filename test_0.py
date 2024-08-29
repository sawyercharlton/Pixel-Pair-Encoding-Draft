# import torch
# import torchvision
# import torchvision.datasets.mnist as mnist
# import os
# from image_minbpe import BasicTokenizer
# from PIL import Image
# from numpy import asarray
#
# torch.set_printoptions(profile="full")  # display the full tensor list
#
# root = "./data/MNIST/raw/"  # change this line according to your own directory.
# train_set = (
#     mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
#     # mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
# )
# # test_set = (
# #     mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
# #     mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
# # )
#
# image_0 = train_set[0][0].view(-1)
# print(image_0)
#
# size = len(image_0[0])
#
# print(30 > size)

# stats = {(0, 0): 518, (0, 1): 13, (0, 2): 4, (0, 3): 7}
# merges = {(0, 0): 256, (0, 1): 279}
# pair = min(stats, key=lambda p: merges.get(p, float("inf")))
# print(pair)

print(5 / 2)