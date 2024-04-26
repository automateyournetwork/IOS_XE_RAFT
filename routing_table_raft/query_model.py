import torch
from peft import LoraConfig, get_peft_model  # Ensure these are correctly imported from your module
from transformers import AutoTokenizer, AutoModelForCausalLM

def attach_lora_adapters(model):
    """Apply LoRA adapters to the model with specified configuration."""
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["Wqkv", "fc1", "fc2"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)

# Load the model and tokenizer from a predefined directory
model_dir = "./phi2-routing-table"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Apply LoRA adapters and print the post-load embedding size
model = attach_lora_adapters(model)
print("Post-load model embedding size:", model.get_input_embeddings().num_embeddings)

# Setup the model on the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()  # Ensure the model is in evaluation mode

def ask_model(question, model, tokenizer, device, max_length=512, num_beams=5):
    """Generate answers to input questions using the model."""
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
