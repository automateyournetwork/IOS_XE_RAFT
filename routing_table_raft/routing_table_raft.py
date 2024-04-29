import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

# Model
base_model = "microsoft/Phi-3-mini-4k-instruct"
new_model = "phi3-routing-table"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model)

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['gate_up_proj', 'down_proj', 'qkv_proj', 'o_proj']
)

file_path = "train_dataset.jsonl"
dataset = raw_dataset = load_dataset('json', data_files={'train': file_path}, split='train')
dataset = dataset.shuffle(seed=42).select(range(253))

def format_for_training(example):
    # Example transformation to create a 'prompt' field
    system_statement = example['messages'][0]['content']  # Assuming the first message is always the system role
    user_question = example['messages'][1]['content']  # Assuming the second message is always the user role
    example['prompt'] = f"{system_statement} {user_question}"
    example['labels'] = example['messages'][2]['content']  # Assuming the third message is always the assistant's response
    return example

# Apply this transformation to each dataset entry
dataset = dataset.map(format_for_training)

dataset = dataset.train_test_split(test_size=0.01)

print(dataset['train'][0])

orpo_args = ORPOConfig(
    learning_rate=8e-6,
    lr_scheduler_type="linear",
    max_length=512,
    max_prompt_length=246,
    beta=0.1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    output_dir=new_model,
    push_to_hub=True
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(new_model)