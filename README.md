# Pixel-Pair-Encoding

1. [pixel_bpe/base.py](pixel_bpe/base.py): Implements the `Tokenizer` class, which is the base class. It contains the `train`, `encode`, and `decode` stubs, save/load functionality, and there are also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
2. [pixel_bpe/basic.py](pixel_bpe/basic.py): Implements the `BasicTokenizer`, the simplest implementation of the BPE algorithm that runs directly on image.


## Quick Start
- run [mnist_vis.py](mnist_vis.py) to download and visualize MNIST dataset.
- run [train.py](train.py) to train a model (vocabulary).
- run [test.py](test.py) to inference.


## Reference
[1]. https://github.com/karpathy/minbpe

## Acknowledgement
Yubo Huang\
Enmao Diao