import torch
from transformers import AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model  # Assuming these are defined as needed

model_dir = "./phi2-routing-table"  # Ensure this is the directory where your fine-tuned model is saved

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load the base model and then apply LoRA adapters as part of the loading process if needed
base_model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
model = get_peft_model(base_model, LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["Wqkv", "fc1", "fc2"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
))

# Ensure model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def ask_model(question, model, tokenizer, device, max_length=512, num_beams=5):
    """Generate answers using the fine-tuned model."""
    inputs = tokenizer.encode_plus(
        question, return_tensors="pt", add_special_tokens=True,
        max_length=max_length, padding="max_length", truncation=True,
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,  # Adjust as needed
            num_beams=num_beams,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Run the model on some example questions
questions = [
    "What can you tell me about my routing table?",
    "What is my default route?",
    "What is the next hop for my default route?",
    "What interface does my default route use?"
]

for question in questions:
    answer = ask_model(question, model, tokenizer, device)
    print(f"Question: {question}\nAnswer: {answer}\n")
