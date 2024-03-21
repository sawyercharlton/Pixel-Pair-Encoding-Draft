# image-minbpe

Modified from [minbpe](https://github.com/sawyercharlton/minbpe/tree/master).
1. [image_minbpe/base.py](image_minbpe/base.py): Implements the `Tokenizer` class, which is the base class. It contains the `train`, `encode`, and `decode` stubs, save/load functionality, and there are also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
2. [image_minbpe/basic.py](image_minbpe/basic.py): Implements the `BasicTokenizer`, the simplest implementation of the BPE algorithm that runs directly on image.

All of the files above are very short and thoroughly commented, and also contain a usage example on the bottom of the file.

You need to change around the vocabulary size depending on the size of your dataset.

## quick start
- run [test.py](test.py).
