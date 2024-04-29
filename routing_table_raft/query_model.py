import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # Ensure PEFT is correctly imported
from dotenv import load_dotenv

def main():   
    model_dir = "./phi2-routing-table"
    base_model = "microsoft/Phi-2"  # Base model for reference if needed

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the base model
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True)
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return

    # Resize token embeddings if necessary
    if len(tokenizer) != model.config.vocab_size:
        print("Resizing token embeddings to match tokenizer's vocabulary size.")
        model.resize_token_embeddings(len(tokenizer))

    # Apply PEFT or load a PEFT model
    try:
        model = PeftModel.from_pretrained(model, model_dir)  # Adjust as necessary
        print("PEFT model loaded successfully.")
    except Exception as e:
        print(f"Failed to load PEFT model: {e}")
        return

    # Save the adjusted model and tokenizer
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    # Example inference to check the model
    questions = [
        "What can you tell me about my routing table?",
        "What is my default route?",
        "What is the next hop for my default route?",
        "What interface does my default route use?"
    ]

    for question in questions:
        answer = ask_model(question, model, tokenizer)
        print(f"Question: {question}\nAnswer: {answer}\n")

def ask_model(question, model, tokenizer, max_length=512, num_beams=5):
    """Generate answers using the fine-tuned model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    main()
