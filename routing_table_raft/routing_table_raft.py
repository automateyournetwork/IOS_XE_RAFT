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

def format_chat_template(example):
    # Iterate through each message and apply formatting or processing
    for message in example['messages']:
        # Check the role and apply specific template or processing
        if message['role'] == 'system':
            message['content'] = f"System says: {message['content']}"
        elif message['role'] == 'user':
            message['content'] = f"User asks: {message['content']}"
        elif message['role'] == 'assistant':
            message['content'] = f"Assistant answers: {message['content']}"
    return example

dataset = dataset.map(
    format_chat_template,
    batched=False,  
    num_proc=os.cpu_count()  

dataset = dataset.train_test_split(test_size=0.01)

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
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(new_model)