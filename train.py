import os
import time
from pixel_bpe import BasicTokenizer
from PIL import Image
from numpy import asarray
import glob
from tqdm import tqdm


in_dir = 'data/MNIST/raw/train'  # change this according to your dataset directory
img_list = []
img_list.extend(glob.glob(os.path.join(in_dir, "*")))
print("training dataset size: ", len(img_list))

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer], ["basic"]):
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    for in_img in tqdm(img_list):
        image = Image.open(in_img)
        img_array = asarray(image)
        img_array_flat = img_array.flatten()
        tokenizer.train(img_array_flat, 256 + 50000, 2, resume=True,
                        verbose=False)  # 256 are the byte tokens, then do ? merges
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)  # writes two files in the models directory: name.model, and name.vocab
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
