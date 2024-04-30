import os
import torch
from datasets import load_dataset
# Disable tokenizer parallelism to avoid potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

# Modela
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

# Load and prepare dataset
file_path = "train_dataset.jsonl"
dataset = load_dataset('json', data_files={'train': file_path}, split='train')
dataset = dataset.shuffle(seed=42).select(range(min(250, len(dataset))))

def format_dataset(entry):
    # Concatenate system and user messages to form the prompt
    prompt = f"{entry['messages'][0]['content']} {entry['messages'][1]['content']}"
    
    # Get the assistant's response for 'chosen'
    chosen = entry['messages'][2]['content']
    
    # Hardcode a generic 'rejected' response
    rejected = "I don't know"
    
    # Return the new structure
    return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}

# Transform the dataset
transformed_dataset = dataset.map(format_dataset, batched=False)  # Ensure batched is False if working with individual records

# Split the transformed dataset for training and testing
split_dataset = transformed_dataset.train_test_split(test_size=0.01)

print(transformed_dataset[0])
print(transformed_dataset[1])

# Configuration for training
orpo_args = ORPOConfig(
    learning_rate=8e-6,
    lr_scheduler_type="linear",
    max_length=512,
    max_prompt_length=246,
    beta=0.1,
    per_device_train_batch_size=25,
    per_device_eval_batch_size=25,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=25,
    logging_steps=25,
    warmup_steps=10,
    output_dir=new_model,
    push_to_hub=False
)

# Initialize the trainer
trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the trained model
trainer.save_model(new_model)