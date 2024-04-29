import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # Make sure this is correct usage
from dotenv import load_dotenv

def main():
    model_dir = "./phi3-routing-table"
    base_model = "microsoft/Phi-3-mini-128k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load the base model
        base_model_instance = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True).to(device)
        print("Base model loaded successfully.")

        # Load the PEFT model, adjust if necessary how PeftModel is supposed to be used
        fine_tuned_model = PeftModel.from_pretrained(model_dir).to(device)  # Assuming model_dir contains fine-tuned model
        print("PEFT model loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    if len(tokenizer) != base_model_instance.config.vocab_size:
        print("Resizing token embeddings to match tokenizer's vocabulary size.")
        base_model_instance.resize_token_embeddings(len(tokenizer))

    # Save the tokenizer and the fine-tuned model (if this is the one you want to keep)
    tokenizer.save_pretrained(model_dir)
    fine_tuned_model.save_pretrained(model_dir)

    questions = [
        "What is my default route, the next hop, and outgoing interface?",
        "What is the next hop for my default route?",
        "What interface does my default route use?"
    ]

    # Testing the models
    test_model("Base Model", base_model_instance, tokenizer, questions, device)
    test_model("Fine-Tuned Model", fine_tuned_model, tokenizer, questions, device)

def test_model(model_name, model, tokenizer, questions, device):
    print(f"\nTesting {model_name}:")
    for question in questions:
        answer = ask_model(question, model, tokenizer, device)
        print(f"Question: {question}\nAnswer: {answer}\n")

def ask_model(question, model, tokenizer, device, max_length=512, num_beams=5):
    system_intro = "You are a computer networking expert specializing in network routing tables."
    prompt = f"{system_intro} User: {question} Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    main()
