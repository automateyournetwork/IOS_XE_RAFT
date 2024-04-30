import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # Ensure PEFT is correctly imported
from dotenv import load_dotenv

def main():   
    model_dir = "./phi3_routing_table"
    base_model = "microsoft/Phi-3-mini-4k-instruct"  # Base model for reference if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the base model
    try:
        base_model_instance = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, trust_remote_code=True)
        base_model_instance = base_model_instance.to(device)
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return

    # Resize token embeddings if necessary
    if len(tokenizer) != base_model_instance.config.vocab_size:
        print("Resizing token embeddings to match tokenizer's vocabulary size.")
        base_model_instance.resize_token_embeddings(len(tokenizer))

    # Apply PEFT or load a PEFT model
    try:
        fine_tuned_model  = PeftModel.from_pretrained(base_model_instance, model_dir)  # Adjust as necessary
        fine_tuned_model = fine_tuned_model.to(device)
        print("PEFT model loaded successfully.")
    except Exception as e:
        print(f"Failed to load PEFT model: {e}")
        return

    # Save the adjusted model and tokenizer
    tokenizer.save_pretrained(model_dir)
    base_model_instance.save_pretrained(model_dir)

    # Example inference to check the model
    questions = [
        "What is my default route?",
        "What is my default route, the next hop, and outgoing interface? Hint: it is 0.0.0.0/0.",
        "What is the next hop for my default route?",
        "What interface does my default route use?",
        "What is the next hop for route 0.0.0.0/0?",
        "What is the outgoing interface for route 0.0.0.0/0?"
    ]

    test_model("Fine-Tuned Model", fine_tuned_model, tokenizer, questions, device)
    test_model("Base Model", base_model_instance, tokenizer, questions, device)

def test_model(model_name, model, tokenizer, questions, device):
    print(f"\nTesting {model_name}:")
    for question in questions:
        answer = ask_model(question, model, tokenizer, device)
        print(f"Question: {question}\nAnswer: {answer}\n")

def ask_model(question, model, tokenizer, device, max_length=512, num_beams=5):
    # Directly use the question as the prompt without any additional text.
    prompt = question

    # Tokenizing the prompt to prepare input for the model.
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Prompt:", prompt)
    print("Tokenized inputs:", inputs)
    
    # Generating the response using the model.
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

    print("Generated outputs:", tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
    # Decoding and returning the response, ensuring special tokens are skipped.
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    main()