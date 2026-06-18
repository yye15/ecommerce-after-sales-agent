# LALUN Integration

This project now has a first working LALUN integration path.

## What Works

The local LALUN research model can be called for English ASTE inference through:

```bash
python scripts\lalun_infer_english.py --text "The bread is top notch as well ."
```

Expected JSON shape:

```json
{
  "enabled": true,
  "engine": "lalun",
  "dataset": "14res",
  "triplets": [
    {
      "aspect": "bread",
      "opinion": "top notch",
      "sentiment": "POS"
    }
  ]
}
```

The LangGraph project calls it through:

```python
from ecommerce_agent.lalun_adapter import LALUNAdapter

adapter = LALUNAdapter()
result = adapter.analyze_english("The service was slow but the food was delicious .")
```

## Why Subprocess

LALUN depends on a research environment:

- `LALUN_PYTHON` points to the Python executable of your local LALUN environment.
- CUDA-enabled PyTorch
- `transformers==4.46.3`
- `pytorch-lightning==1.3.5`

The main Agent project uses a separate LangChain/LangGraph environment. Calling LALUN through a subprocess keeps both dependency stacks isolated.

## Current Limitation

This is not yet Chinese e-commerce inference.

Current support:

- English sentences
- ASTE-style aspect-opinion-sentiment triplets
- Best tested with restaurant/laptop style English reviews

Still pending:

- Chinese tokenizer/POS/dependency preprocessing
- Chinese ASTE training data
- Chinese LALUN fine-tuning
- Replacing the current DeepSeek/rule sentiment node with LALUN for Chinese production use

## Verified Examples

```text
The bread is top notch as well .
=> bread / top notch / POS

The service was slow but the food was delicious .
=> service / slow / NEG
=> food / delicious / POS

The waiter was rude .
=> waiter / rude / NEG
```
