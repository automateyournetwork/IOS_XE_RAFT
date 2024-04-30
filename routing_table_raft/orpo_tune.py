import gc
import os
from datasets import load_dataset, Dataset
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

# Initialize wandb
wandb.login()
os.environ["WANDB_PROJECT"] = "phi3-finetune" if "phi3-finetune" else ""

if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# Model
base_model = "microsoft/Phi-3-mini-4k-instruct"
new_model = "phi3_routing_table"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['gate_up_proj', 'down_proj', 'qkv_proj', 'o_proj']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

def transform_dataset(dataset):
    transformed_data = {'prompt': [], 'chosen': [], 'rejected': []}
    for row in dataset:
        messages = row["messages"]
        user_message = next((msg["content"] for msg in messages if msg.get("role") == "user"), None)
        assistant_message = next((msg["content"] for msg in messages if msg.get("role") == "assistant"), None)
        
        if user_message and assistant_message:
            transformed_data['prompt'].append(user_message)
            transformed_data['chosen'].append(assistant_message)
            transformed_data['rejected'].append("I don't know")
    
    return transformed_data

def format_chat_template(row):
    if isinstance(row, dict):
        row["chosen"] = tokenizer.apply_chat_template(row.get("chosen", ""), tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row.get("rejected", "I don't know"), tokenize=False)
    else:
        print("Warning: Unexpected input format in format_chat_template function:", row)
    return row

# Load the dataset
file_path = "train_dataset.jsonl"
dataset = load_dataset('json', data_files={'train': file_path}, split='train')
dataset = dataset.shuffle(seed=42).select(range(253))

# Transform dataset
transformed_dataset = transform_dataset(dataset)

# Convert the structured data to a Dataset object
formatted_dataset = Dataset.from_dict(transformed_dataset)

# Apply format_chat_template function
formatted_dataset = formatted_dataset.map(
    format_chat_template,
    num_proc=os.cpu_count(),
    batched=True  # Assuming batch processing if necessary
)

# Split the dataset
formatted_dataset = formatted_dataset.train_test_split(test_size=0.01)


orpo_args = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    report_to="wandb",
    output_dir="./results/",
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(new_model)

# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()

# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model, tokenizer = setup_chat_format(model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()