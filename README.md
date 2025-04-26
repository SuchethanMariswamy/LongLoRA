# LongLoRA Fine-Tuning

## Overview

This repository provides scripts and instructions to fine-tune Large Language Models (LLMs) using the LongLoRA approach, enabling extended context lengths efficiently.

## Table of Contents

- Installation
- Dataset Preparation
- Training
- Inference
- Evaluation
- Citation

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/longlora-finetune.git
cd longlora-finetune
```

2. Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Dataset Preparation

The fine-tuning process requires a dataset in JSON format with the following structure:

```json
[
  {
    "instruction": "Translate English to French",
    "input": "How are you?",
    "output": "Comment ça va ?"
  },
  {
    "instruction": "Summarize the following paragraph.",
    "input": "Large Language Models are very powerful...",
    "output": "Summary of LLM capabilities."
  }
]
```

Save this as `data/longalpaca.json` in the project directory.

## Training

To fine-tune the model using LongLoRA:

```bash
python train_longlora.py \
    --model_name_or_path TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0 \
    --data_path ./data/longalpaca.json \
    --output_dir ./output/finetuned_model \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --fp16 True \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2
```

### Parameters:

- `model_name_or_path`: Pretrained model to fine-tune.
- `data_path`: Path to the training dataset.
- `output_dir`: Directory to save the fine-tuned model.
- `per_device_train_batch_size`: Batch size per device.
- `gradient_accumulation_steps`: Number of gradient accumulation steps.
- `num_train_epochs`: Number of training epochs.
- `learning_rate`: Learning rate for training.
- `fp16`: Use 16-bit (mixed) precision training.
- `logging_steps`: Log every X updates steps.
- `save_steps`: Save checkpoint every X updates steps.
- `save_total_limit`: Maximum number of checkpoints to save.

## Inference

After training, you can generate responses using the fine-tuned model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./output/finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("./output/finetuned_model")

prompt = "What are the benefits of using transformer models?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Evaluation

To evaluate the model's performance, you can use standard NLP metrics such as BLEU, ROUGE, or perplexity.  
Implement evaluation scripts as needed based on your specific use case and dataset.


