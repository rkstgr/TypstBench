# TypstBench

LLM evaluation suite for Typst.

TypstBench is a dataset containing tasks and question around Typst, a modern typesetting language. It serves as a reference point for evaluating and enhancing LLMs' proficiency in Typst generation.

## Preliminary Findings

| Model                  | Accuracy   |
|------------------------|------------|
| Gemini 2.5 Pro         | 65.22%     |
| Claude 3.7 Sonnt       | 60.87%     |
| Claude 4.5 Haiku       | 56.52%     |
| Gemini 2.5 Flash       | 56.52%     |
| GPT-4.1                | 21.74%     |
| GPT-4.1-Mini           | 8.70%      |

## Statistics

1. Get statistics of the dataset
```sh
python dataset.py stats
```

## Verify

This renders all task solutions using typst, ensuring the solutions are syntactically correct.
Note: Some tasks are ignored, either by having a `.ignore-verify` in their category directory (like 'multiple-choice'), or if the flag `ignore-verify = true` is set in the frontmatter of the task definition.
```sh
python dataset.py verify
```

## Render

Render dataset samples to pdf to inspect
```sh
# render a specific task
python dataset.py render generate/001

# render all tasks (that are not filtered out)
python dataset.py render --all
```

## Evaluate

Evaluate an external model using LiteLLM against all samples. I would advise creating a local .envrc file containing the API keys to the LLM providers.
```sh
# Basic usage with OpenAI GPT-3.5
python evaluate.py --model gpt-3.5-turbo --api-key your_openai_key

# Evaluate with Claude on basic tier samples only
python evaluate.py --model anthropic/claude-3-opus-20240229 --api-key your_anthropic_key --tier basic

# Evaluate only math-related samples with a maximum of 5 samples
python evaluate.py --model gpt-4 --features math --max-samples 5

# Reduce the concurrency
python evaluate.py --model anthropic/claude-3-5-haiku-20241022 --concurrency 2
```
