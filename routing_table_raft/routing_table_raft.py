from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
NEW_MODEL_NAME = "phi3-routing-table"
DATASET_PATH = "train_dataset.jsonl"
MAX_SEQ_LENGTH = 2048
num_train_epochs = 1
learning_rate = 1.41e-5
per_device_train_batch_size = 4
gradient_accumulation_steps = 1

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")

EOS_TOKEN=tokenizer.eos_token_id

def process_dataset(mydata):
    texts = []
    for entry in mydata["messages"]:
        role_mapper = {"system": "system\n", "user": "\nuser\n", "assistant": "\nassistant\n"}
        text = "".join(f"{role_mapper[x['role']]} {x['content']}\n" for x in entry)
        texts.append(f"{text}{EOS_TOKEN}")
    return {"text": texts}

dataset = dataset.map(process_dataset, batched=True)

print(dataset['text'][2])

args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=num_train_epochs,
        save_strategy="no",
        logging_steps=1,
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

trainer.train()
