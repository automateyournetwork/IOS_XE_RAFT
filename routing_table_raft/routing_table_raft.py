from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import torch

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
NEW_MODEL_NAME = "phi3-routing-table"
DATASET_PATH = "train_dataset.jsonl"
MAX_SEQ_LENGTH = 1024  # Adjusted if possible for your context
num_train_epochs = 1
learning_rate = 1.41e-5
per_device_train_batch_size = 2  # Reduced batch size
gradient_accumulation_steps = 8  # Increased gradient accumulation

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU."

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")

EOS_TOKEN = tokenizer.eos_token_id

def process_dataset(mydata):
    texts = []
    for entry in mydata["messages"]:
        role_mapper = {"system": "system\n", "user": "\nuser\n", "assistant": "\nassistant\n"}
        text = "".join(f"{role_mapper[x['role']]} {x['content']}\n" for x in entry)
        texts.append(f"{text}{EOS_TOKEN}")
    return {"text": texts}

dataset = dataset.map(process_dataset, batched=True)

args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    num_train_epochs=num_train_epochs,
    save_strategy="no",
    logging_steps=100,  # Adjusted logging steps to reduce verbosity
    output_dir=NEW_MODEL_NAME,
    optim="paged_adamw_32bit",
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    formatting_func=process_dataset
)

# Optional: Clear CUDA cache before starting training
torch.cuda.empty_cache()

# Start training
trainer.train()
