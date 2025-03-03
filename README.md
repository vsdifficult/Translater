# Neural Machine Translator

A neural network project for translating text between different languages using the Transformer architecture.

## Repository Structure

```
Translator/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   └── translate.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_tokenizer.py
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/vsdifficult/Translator.git
cd Translator
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. For Simple Text Translation

```python
from neural_translator import Translator

translator = Translator(source_lang="en", target_lang="ru")
translation = translator.translate("Hello, how are you?")
print(translation)  # "Привет, как дела?"
```

### 2. For Training a New Model

```bash
python src/train.py --config configs/ru_en_config.json
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- transformers
- numpy
- tqdm