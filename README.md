# EN2DE-Transformer

An English-to-German neural machine translation model: a Transformer implemented from scratch in TensorFlow 2, following "Attention Is All You Need" (Vaswani et al., 2017).

## Overview

The whole pipeline lives in a single notebook, `model.ipynb`: text cleaning, subword tokenization, the Transformer architecture, training, and interactive translation. It is trained on the Europarl English-German parallel corpus.

## Features

- Transformer built from custom Keras layers: positional encoding, scaled dot-product attention, multi-head attention, encoder and decoder stacks
- Padding and look-ahead masks
- Subword tokenization (`SubwordTextEncoder`, 8,000 target vocabulary per language)
- Custom warm-up learning-rate schedule from the paper, masked loss, checkpointing
- Greedy decoding for translating new sentences

## Project structure

```
.
├── model.ipynb        # data prep, model, training, evaluation
├── requirements.txt
├── LICENSE
└── data/              # not tracked; see Data below
```

## Setup

```bash
git clone https://github.com/m0han-raj/EN2DE-Transformer.git
cd EN2DE-Transformer
pip install -r requirements.txt
```

`tfds.deprecated.text.SubwordTextEncoder` is only available in older `tensorflow-datasets` releases; if the import fails, install a version that still ships it.

### Data

Create a `data/` folder containing:

- `europarl-v7.de-en.en` and `europarl-v7.de-en.de` from the [Europarl corpus](https://www.statmt.org/europarl/)
- `nonbreaking_prefix.en` and `nonbreaking_prefix.de` from the [Moses non-breaking prefixes](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/share/nonbreaking_prefixes)

## Usage

Open `model.ipynb` and run the cells in order from the repository root. Hyperparameters (`D_MODEL=512`, 6 layers, 2048 FFN units, 8 heads, dropout 0.1, `MAX_LENGTH=20`, batch size 64) are set in the notebook. The notebook sets `EPOCHS = 3`; the original notes suggest 3 epochs on CPU and 5 on GPU. The last cell prompts for a sentence and prints the translation.

## Results

The notebook has no quantitative evaluation. One saved example from a short CPU training run:

```
Input: how are you?
Predicted translation: Wie sind Sie das tun?
```

Sentences are capped at 20 tokens and training is brief, so translations are rough.

## Tech stack

Python, TensorFlow 2 / Keras, TensorFlow Datasets, NumPy

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017
- [Europarl parallel corpus](https://www.statmt.org/europarl/)
- [Moses decoder](https://github.com/moses-smt/mosesdecoder) non-breaking prefixes

## License

MIT, see [LICENSE](LICENSE).
