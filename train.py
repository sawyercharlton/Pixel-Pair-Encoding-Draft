"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from image_minbpe import BasicTokenizer
from PIL import Image
from numpy import asarray


# open some text and train a vocab of 256+? tokens
image = Image.open('./0.jpg')
img_np = asarray(image)
img_np_flat = img_np.flatten()
text = img_np_flat

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer], ["basic"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 256 + 48, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")