# TypstBench


Benchmark dataset to evaluate LLM performance to write typst markdown

1. Get statistics of the dataset
```sh
python dataset.py
```

2. Render dataset samples to pdf to inspect
```sh
# render all
python render.py

# Keep the typ source files
python render.py --keep-typ
```

2. Evaluate an external model using LiteLLM
```sh
# Basic usage with OpenAI GPT-3.5
python evaluate.py --model gpt-3.5-turbo --api-key your_openai_key

# Evaluate with Claude on basic tier samples only
python evaluate.py --model anthropic/claude-3-opus-20240229 --api-key your_anthropic_key --tier basic

# Evaluate only math-related samples with a maximum of 5 samples
python evaluate.py --model gpt-4 --features math --max-samples 5
```
