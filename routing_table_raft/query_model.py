import torch
from peft import LoraConfig, get_peft_model, PeftModel  # Ensure these are correctly imported from your module
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_base_model():
    # Load a pre-trained model or initialize it
    return AutoModelForCausalLM.from_pretrained("microsoft/phi-2", local_files_only=True)

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
# Load the tokenizer with any special tokens added during training
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Ensure you save the tokenizer if you modify it
    tokenizer.save_pretrained(model_dir)

# Load the base model
base_model = load_base_model()

# Attach LoRA adapters
model_with_lora = attach_lora_adapters(base_model)

# Assuming you need to save the modified model with LoRA adapters
model_with_lora.save_pretrained(model_dir)  # Save the adapted model

# Attempt to load the model
try:
    model = PeftModel.from_pretrained(model=model_dir, model_id=model_dir)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {str(e)}")

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
